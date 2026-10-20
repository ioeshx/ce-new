"""Shared edit groups, absolute weight switching and checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def candidate_groups(unet, params: str) -> dict[str, dict]:
    """One selection unit per attention layer; K/V are removed together.
        returns {group_name: {"parent": module, "weights": {param_name: tensor}}}
    """
    groups = {}
    for name, parent in unet.named_modules():
        if not name.endswith("attn2"):
            continue
        weights = {}
        for letter in params.lower():
            linear = getattr(parent, f"to_{letter}", None)
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(f"{name}.to_{letter} is not an unfused linear layer.")
            weights[f"{name}.to_{letter}.weight"] = linear
        groups[name] = {"parent": parent, "weights": weights}
    if not groups:
        raise ValueError("No SD cross-attention K/V candidate modules were found.")
    return groups


@torch.no_grad()
def actual_kv_input(parent, hidden: torch.Tensor) -> torch.Tensor:
    """Exactly the input used by standard SD AttnProcessor/AttnProcessor2_0.

    SD1/SDXL K/V inputs are text states (optionally norm_cross), independent of
    image latents/timestep. A U-Net sampling pass is consequently unnecessary.
    This shortcut is NOT valid for evolving text states in a DiT.
    """
    if not getattr(parent, "norm_cross", False):
        return hidden
    parameter = next(parent.parameters())
    h = hidden.unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
    return parent.norm_encoder_hidden_states(h)[0].float().cpu()


class EditController:
    """Apply absolute original/edited values; never accumulate patches."""

    def __init__(self, model, original, edited, groups, architecture="sd1"):
        self.parameters = dict(model.named_parameters())
        self.original = original
        self.edited = edited
        self.groups = groups
        self.architecture = architecture
        self.active: set[str] = set()
        for key, value in edited.items():
            if key not in self.parameters or self.parameters[key].shape != value.shape:
                raise ValueError(f"Checkpoint does not match model parameter: {key}")

    @torch.no_grad()
    def apply(self, active):
        active = set(active)
        if active - self.groups.keys():
            raise ValueError(f"Unknown groups: {sorted(active - self.groups.keys())}")

        enabled_keys = {key for group in active for key in self.groups[group]}
        for key in self.edited:
            tensor = self.edited[key] if key in enabled_keys else self.original[key]
            self.parameters[key].copy_(tensor)
        self.active = active

    def save(self, directory: Path, metadata: dict):
        directory.mkdir(parents=True, exist_ok=True)
        # save weights only for active groups, to reduce checkpoint size
        active_keys = {key for group in self.active for key in self.groups[group]}
        weights = {k: self.edited[k].contiguous().cpu() for k in sorted(active_keys)}
        save_file(weights, str(directory / "weights.safetensors"))
        # save info
        info = {
            **metadata, "format": f"casu2-{self.architecture}-v1",
            "architecture": self.architecture,
            "active_groups": sorted(self.active), "groups": self.groups,
        }
        (directory / "metadata.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )


def load_controller(model, directory: Path, architecture="sd1"):
    directory = directory.parent if directory.is_file() else directory
    # load info
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != f"casu2-{architecture}-v1":
        raise ValueError(f"Not a CASU-2 {architecture} checkpoint.")
    # load weights
    weights = load_file(str(directory / "weights.safetensors"), device="cpu")
    params = dict(model.named_parameters())
    originals = {}
    for key, value in weights.items():
        if key not in params or params[key].shape != value.shape:
            raise ValueError(f"Incompatible base model for {key}")
        originals[key] = params[key].detach().cpu().clone()
    groups = {
        group: keys for group, keys in metadata["groups"].items()
        if group in metadata["active_groups"]
    }

    expected_keys = {key for keys in groups.values() for key in keys}
    if expected_keys != set(weights):
        raise ValueError("Checkpoint tensor keys do not match active groups.")
    return EditController(model, originals, weights, groups, architecture), metadata
