# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, build_ffn
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY

try:
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
except:
    unpad_input, pad_input = None, None

attn_config_defaults: Dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'triton',
    'qk_ln': False,
    'qk_gn': False,
    'clip_qkv': None,
    'softmax_scale': None,
    'prefix_lm': False,
    'attn_uses_sequence_id': False,
    'sliding_window_size': -1,
    'alibi': False,
    'alibi_bias_max': 8,
    'rope': False,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}


class MPTBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        ffn_config: Optional[Dict] = None,
        resid_pdrop: float = 0.0,
        residual_modification: Optional[str] = None,
        residual_coeff: Optional[float] = None,
        norm_type: str = 'low_precision_layernorm',
        fc_type: str = 'torch',
        device: Optional[str] = None,
        no_bias: bool = False,
        layer_num: int = 1,
        use_pad_tok_in_ffn: bool = True,
        **kwargs: Any,
    ):
        if attn_config is None:
            attn_config = attn_config_defaults

        if ffn_config is None:
            ffn_config = {
                'ffn_type': 'mptmlp',
            }

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        assert isinstance(attn_config['attn_type'], str)
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

        # necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
        args_to_exclude_in_attn_class = {
            'attn_type', 'prefix_lm', 'alibi', 'attn_uses_sequence_id',
            'alibi_bias_max', 'rope', 'rope_theta', 'rope_impl',
            'rope_dail_config', 'rope_hf_config'
        }
        attn_config_subset_for_attn_class = {
            k: v
            for k, v in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }

        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            fc_type=fc_type,
            device=device,
            **attn_config_subset_for_attn_class,
            bias=not no_bias,
        )
        self.norm_2 = None
        if not getattr(FFN_CLASS_REGISTRY[ffn_config['ffn_type']], '_has_norm',
                       False):
            self.norm_2 = norm_class(d_model, device=device)
        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            **ffn_config,
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

        self.residual_modification = residual_modification
        if self.residual_modification is not None and \
            self.residual_modification not in ['coefficient', 'layer']:
            raise ValueError(f"If using residual_modification, must be set to either ",
                             f"'coefficient' or 'layer'. Got {self.residual_modification} ",
                             f"instead.")
        if self.residual_modification == 'layer':
            self.branch_coeff_attn = 1/np.sqrt(2*layer_num)
            self.stream_coeff_attn = np.sqrt((2*layer_num-1)/(2*layer_num))
            self.branch_coeff_ffn = 1/np.sqrt(2*layer_num+1)
            self.stream_coeff_ffn = np.sqrt((2*layer_num)/(2*layer_num+1))
        elif self.residual_modification == 'coefficient':
            self.branch_coeff_attn = np.sqrt(residual_coeff)
            self.stream_coeff_attn = np.sqrt(1-residual_coeff)
            self.branch_coeff_ffn = np.sqrt(residual_coeff)
            self.stream_coeff_ffn = np.sqrt(1-residual_coeff)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[Dict] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[
            torch.Tensor, torch.Tensor]]]:
        a, attn_weights, past_key_value = self.attn(
            x,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            rotary_emb_w_meta_info=rotary_emb_w_meta_info,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
            alibi_slopes=alibi_slopes,
            flash_attn_padding_info=flash_attn_padding_info,
        )
        # Moved first layer norm after attention and before residual connection.
        b = self.norm_1(a)
        if self.residual_modification is not None:
            x = self.stream_coeff_attn*x + self.branch_coeff_attn*self.resid_attn_dropout(b)
        else:
            x = x + self.resid_attn_dropout(b)
        m = x
        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn:
            assert unpad_input is not None
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        # Moved second layer norm after FFN and before residual connection.
        if self.norm_2 is not None:
            n = self.norm_2(n)
        if self.residual_modification is not None:
            x = self.stream_coeff_ffn*x + self.branch_coeff_ffn*self.resid_attn_dropout(b)
        else:
            x = x + self.resid_attn_dropout(b)
        return x, attn_weights, past_key_value
