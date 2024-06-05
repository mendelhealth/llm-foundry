# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Enable curriculum learning by resuming with a different dataset.

This callback is currently experimental. The API may change without warning in
the future.
"""

import copy
import logging
from typing import Any, Union

from composer import DataSpec
from composer.core import State, Time, TimeUnit, ensure_time
from composer.loggers import Logger, MosaicMLLogger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from torch.utils.data import DataLoader

from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils.exceptions import (
    BaseContextualError,
    TrainDataLoaderLocation,
)

log = logging.getLogger(__name__)

__all__ = ['CurriculumLearning']


class CurriculumLearning(CallbackWithConfig):
    """Starts an epoch with a different dataset when resuming from a checkpoint.

    Args:
        train_config (Dict): The configuration of the dataset currently
            being used. Note that this is the full train config and must
            contain the 'train_loader' key.
        dataset_index (int): The index of the dataset currently being used.
    """

    def __init__(
        self,
        train_config: dict[str, Any],
        schedule: list[dict[str, Any]],
    ):
        from llmfoundry.utils.builders import build_tokenizer
        from llmfoundry.utils.config_utils import calculate_batch_size_info

        # Ensure all duration values are positive
        # Ensure all duration units are in epochs or tokens
        self._schedule = schedule
        if len(self._schedule) == 0:
            raise ValueError('The schedule must have at least one datamix.')
        for index, datamix in enumerate(self._schedule):
            self._validate_data_mix(datamix)

            if (
                index == 0 and
                train_config['train_loader'] != datamix['train_loader']
            ):
                raise ValueError((
                    'The first datamix in the schedule must match the '
                    'train_loader in the train_config.'
                ))

        self._schedule_index = 0

        # Copied from llmfoundry/utils/config_utils.py
        self.device_train_batch_size, _, _ = calculate_batch_size_info(
            train_config['global_train_batch_size'],
            train_config['device_train_microbatch_size'],
            data_replication_degree=1,
        )

        # Copied from scripts/train/train.py
        tokenizer_name = train_config['tokenizer']['name']
        tokenizer_kwargs = train_config['tokenizer'].get('kwargs', {})
        self.tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    def before_load(self, state: State, logger: Logger):
        del logger

        # Ensure all duration units are the same as max_duration
        units_match = True
        assert state.max_duration is not None, 'max_duration should have beeen set.'
        for datamix in self._schedule:
            if datamix['duration'].unit != state.max_duration.unit:
                units_match = False
        if not units_match:
            raise ValueError((
                'All durations in the schedule must have the same units as '
                'the max_duration.'
            ))

        # Ensure schedule duration is equal to max_duration
        schedule_duration = Time(0, state.max_duration.unit)
        for datamix in self._schedule:
            assert isinstance(datamix['duration'], Time)
            schedule_duration += datamix['duration']
        if schedule_duration != state.max_duration:
            raise ValueError((
                'The sum of all durations in the schedule must be equal to the '
                'max_duration.'
            ))

        self._validate_dataloader(state.train_dataloader)

    def after_load(self, state: State, logger: Logger):
        del logger

        self._validate_dataloader(state.train_dataloader)

        # If checkpoint was saved before iteration was incremented, we need to increment it now
        if ((
            self._schedule[self._schedule_index]['duration'].unit
            == TimeUnit.TOKEN and state.timestamp.token_in_iteration
            >= self._schedule[self._schedule_index]['duration'].value
        ) or (
            self._schedule[self._schedule_index]['duration'].unit
            == TimeUnit.EPOCH and state.timestamp.epoch_in_iteration
            >= self._schedule[self._schedule_index]['duration'].value
        )):
            log.error((
                'The CurriculumLearning callback has detected that the previous run did not correctly '
                'increment the iteration.'
            ))
            self._schedule_index += 1
            state.timestamp = state.timestamp.to_next_iteration()

    def iteration_start(self, state: State, logger: Logger):
        # Reset and initialize state train dataloader
        log.warning(
            'trainer._train_data_spec should be updated whenever the dataloader is updated',
        )

        # Swap the dataset if starting a new iteration that's not the original datamix
        if self._schedule_index > 0:
            clean_stale_shared_memory()
            datamix = copy.deepcopy(self._schedule[self._schedule_index])
            data_spec = self._build_train_loader(
                train_loader_config=datamix['train_loader'],
                logger=logger,
            )
            state.set_dataloader(
                dataloader=data_spec.dataloader,
                dataloader_label='train',
            )
            # state.train_dataloader = state.dataloader # FIXME
            state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
                state.device_train_microbatch_size,
                state.auto_microbatching,
                state.train_dataloader,
            )
            self._validate_dataloader(state.train_dataloader)

        # Set the length of the new iteration
        state._iteration_length = self._schedule[self._schedule_index
                                                ]['duration']

    def iteration_end(self, state: State, logger: Logger):
        del state, logger  # unused

        self._schedule_index += 1

    def state_dict(self):
        return {
            'schedule': self._schedule,
            'schedule_index': self._schedule_index,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self._schedule_index = state['schedule_index']

        # Ensure that the schedule has not changed on previously trained datamixes
        for idx in range(state['schedule_index']):
            assert self._schedule[idx] == state['schedule'][idx]

        # Ensure that the datamix has not changed on the current datamix
        current_loader = self._schedule[self._schedule_index]['train_loader']
        saved_loader = state['schedule'][self._schedule_index]['train_loader']
        assert current_loader == saved_loader

        # Ensure that the current datamix duration is greater than timestamp
        duration = self._schedule[self._schedule_index]['duration']
        assert isinstance(duration.unit, (TimeUnit.TOKEN, TimeUnit.EPOCH))
        if ((
            duration.unit == TimeUnit.TOKEN and
            duration > state['timestamp'].token_in_iteration
        ) or (
            duration.unit == TimeUnit.EPOCH and
            duration > state['timestamp'].epoch_in_iteration
        )):
            raise ValueError((
                'The duration of the current datamix must be less or equal to '
                'than the saved timestamp.'
            ))

    def _build_train_loader(
        self,
        train_loader_config: dict[str, Any],
        logger: Logger,
    ) -> DataSpec:
        from llmfoundry.data.dataloader import build_dataloader

        # Copied from scripts/train/train.py
        log.info(
            f'Building train loader in CurriculumLearning callback for dataset {self._schedule_index}',
        )
        try:
            return build_dataloader(
                train_loader_config,
                self.tokenizer,
                self.device_train_batch_size,
            )
        except BaseContextualError as e:
            for destination in logger.destinations:
                if isinstance(destination, MosaicMLLogger):
                    e.location = TrainDataLoaderLocation
                    destination.log_exception(e)
            raise e

    def _validate_dataloader(self, train_loader: Any):
        # Check if we are using a DataLoader and StreamingDataset
        if not isinstance(train_loader, DataLoader):
            raise ValueError(
                f'CurriculumLearning callback can only be used with a train ',
                f'dataloader of type DataLoader, but got {type(train_loader)}.',
            )
        dataset = train_loader.dataset
        if not isinstance(dataset, StreamingDataset):
            raise ValueError(
                f'CurriculumLearning callback only supports StreamingDataset ',
                f'because it requires loading and saving dataset state. ',
                f'Instead, got a dataset of type {type(dataset)}',
            )

    def _validate_datamix(self, datamix: dict[str, Any]):
        if 'duration' not in datamix:
            raise ValueError('Each datamix must have a duration.')
        datamix['duration'] = ensure_time(
            datamix['duration'],
            TimeUnit.EPOCH,
        )
        if datamix['duration'].value <= 0:
            raise ValueError('The duration must be positive.')
        if (
            datamix['duration'].unit != TimeUnit.EPOCH and
            datamix['duration'].unit != TimeUnit.TOKEN
        ):
            raise ValueError(
                'Schedules can only be defined in terms of epochs or tokens.',
            )

        if 'train_loader' not in datamix:
            raise ValueError('Each datamix must have a train_loader.')
