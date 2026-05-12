#!/usr/bin/env python

import torch

from lerobot.policies.vlsarm.modeling_vlsarm import VLSARMRewardModel


def test_vlsarm_pools_image_tokens_before_falling_back_to_last_token():
    model = object.__new__(VLSARMRewardModel)
    torch.nn.Module.__init__(model)
    model.vl_image_token_ids = (42,)

    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [10.0, 10.0],
            ],
            [
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
                [99.0, 99.0],
            ],
        ]
    )
    model_inputs = {
        "input_ids": torch.tensor(
            [
                [10, 42, 42, 11],
                [10, 12, 13, 14],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ]
        ),
    }

    pooled = model._pool_visual_tokens(hidden_states, model_inputs)

    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(pooled[1], torch.tensor([7.0, 7.0]))
