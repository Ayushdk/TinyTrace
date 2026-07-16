import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_tinytrace import (
    build_optimizer,
    rebuild_optimizer_preserving_state,
    save_checkpoint,
)
from tinytrace.config import TinyTraceConfig
from tinytrace.model import TinyTraceModel

from test_vision import FakeMobileCLIPBackbone


class TrainingStateTests(unittest.TestCase):
    def test_stage_transition_preserves_existing_adam_state(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
        tracked_parameter = model.text_head.weight
        tracked_parameter.square().mean().backward()
        optimizer.step()
        previous_step = optimizer.state[tracked_parameter]["step"].clone()

        model.set_visual_encoder_trainable(True, strategy="conv_exp")
        rebuilt = rebuild_optimizer_preserving_state(
            model,
            optimizer,
            lr=1e-3,
            weight_decay=0.01,
            visual_lr_scale=0.1,
        )

        self.assertEqual(len(rebuilt.param_groups), 2)
        self.assertTrue(torch.equal(rebuilt.state[tracked_parameter]["step"], previous_step))
        visual_parameters = {
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("visual_encoder.mobileclip.") and parameter.requires_grad
        }
        self.assertTrue(visual_parameters)
        self.assertTrue(
            visual_parameters.issubset(set(rebuilt.param_groups[1]["params"]))
        )

    def test_stage2_checkpoint_restores_matching_optimizer_layout(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        model.set_visual_encoder_trainable(True, strategy="conv_exp")
        optimizer = build_optimizer(
            model,
            lr=1e-3,
            weight_decay=0.01,
            visual_lr_scale=0.1,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stage2.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                epoch=2,
                best_loss=1.0,
                history=[],
                training_state={
                    "stage2_activated": True,
                    "stage2_unfreeze_strategy": "conv_exp",
                    "stage2_visual_lr_scale": 0.1,
                },
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        restored = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        restored.load_state_dict(checkpoint["model_state"])
        restored.set_visual_encoder_trainable(
            True,
            strategy=checkpoint["training_state"]["stage2_unfreeze_strategy"],
        )
        restored_optimizer = build_optimizer(
            restored,
            lr=1e-3,
            weight_decay=0.01,
            visual_lr_scale=checkpoint["training_state"]["stage2_visual_lr_scale"],
        )
        restored_optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.assertEqual(len(restored_optimizer.param_groups), 2)
        self.assertTrue(checkpoint["training_state"]["stage2_activated"])


if __name__ == "__main__":
    unittest.main()
