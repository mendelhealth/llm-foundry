# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf as om

from llmfoundry.utils.builders import build_callback


def test_curriculum_learning_callback_builds():
    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    kwargs = {
        'schedule': [{
            'duration': '1ep',
            'train_loader': test_cfg.train_loader
        }, {
            'duration': '2ep',
            'train_loader': {}
        }]
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    assert callback is not None
