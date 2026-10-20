"""SD1.x adapter: text-only cross-attention K/V inputs and model loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from prompts import PromptPair, aligned_positions
from solver import solve_transform
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from safetensors.torch import save_file, load_file

def load_pipeline(model: str, device: str, dtype: str, local_only: bool = False):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU environment or explicitly --device cpu.")
    if dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    if device == "cpu" and dtype == "float16":
        raise ValueError("Use --dtype float32 for CPU inference.")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model, torch_dtype=getattr(torch, dtype), local_files_only=local_only,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    
    cross_dim = pipe.unet.config.cross_attention_dim
    cross_dims = cross_dim if isinstance(cross_dim, (tuple, list)) else [cross_dim]
    if int(pipe.text_encoder.config.hidden_size) != 768 or set(cross_dims) != {768}:
        raise ValueError("This implementation targets SD1.x (768-D CLIP). SDXL/SD2/SD3/FLUX need adapters.")
    # if getattr(pipe.unet.config, "addition_embed_type", None):
    #     raise ValueError("Additional pooled-text conditioning needs a separate architecture adapter.")
    # if getattr(pipe.unet.config, "encoder_hid_dim_type", None):
    #     raise ValueError("This checkpoint transforms encoder inputs before attention; adapter unsupported.")
    return pipe


@dataclass
class EncodedText:
    hidden: torch.Tensor
    ids: list[int]
    length: int


class TextCache:
    def __init__(self, pipe):
        self.pipe = pipe
        self.cache: dict[str, EncodedText] = {}

    @torch.no_grad()
    def get(self, text: str) -> EncodedText:
        if text not in self.cache:
            tok = self.pipe.tokenizer
            raw = tok(text, truncation=False, add_special_tokens=True)["input_ids"]
            if len(raw) > tok.model_max_length:
                raise ValueError(f"Prompt is truncated by CLIP; shorten it: {text!r}")
            inputs = tok(
                text, padding="max_length", max_length=tok.model_max_length,
                truncation=True, return_tensors="pt",
            )
            kwargs = {}
            device = next(self.pipe.text_encoder.parameters()).device
            if getattr(self.pipe.text_encoder.config, "use_attention_mask", False):
                kwargs["attention_mask"] = inputs.attention_mask.to(device)
            hidden = self.pipe.text_encoder(inputs.input_ids.to(device), **kwargs)[0]
            self.cache[text] = EncodedText(
                hidden[0].detach().float().cpu(), 
                inputs.input_ids[0].tolist(),
                int(inputs.attention_mask[0].sum()),
            )
        return self.cache[text]


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

    SD1.x K/V inputs are text states (optionally norm_cross), independent of
    image latents/timestep. A U-Net sampling pass is consequently unnecessary.
    This shortcut is NOT valid for evolving text states in a DiT.
    """
    if not getattr(parent, "norm_cross", False):
        return hidden
    parameter = next(parent.parameters())
    h = hidden.unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
    return parent.norm_encoder_hidden_states(h)[0].float().cpu()


def module_inputs(
    parent, cache: TextCache, pairs: list[PromptPair], retains: list[str],
    token_mode: str, include_padding: bool, protect_null: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build K/V inputs for all cross-attention layers in a U-Net module.
        returns (xt, xa, xp) for target, anchor, and retained concepts.
    """
    source, anchor, protection = [], [], []
    for pair in pairs:
        target, reference = cache.get(pair.target_prompt), cache.get(pair.anchor_prompt)
        ti, ai = aligned_positions(
            target.ids, reference.ids, target.length, reference.length,
            token_mode, include_padding,
        )
        source.append(actual_kv_input(parent, target.hidden)[ti])
        anchor.append(actual_kv_input(parent, reference.hidden)[ai])
    # Preserve last SUBJECT tokens of unrelated concepts, as explicit samples.
    for text in retains:
        encoded = cache.get(text)
        protection.append(actual_kv_input(parent, encoded.hidden)[max(1, encoded.length - 2):encoded.length - 1])
    if protect_null:
        # SD passes padding states to cross-attention too. Protect all null states.
        protection.append(actual_kv_input(parent, cache.get("").hidden))
    xt, xa = torch.cat(source), torch.cat(anchor)
    xp = torch.cat(protection) if protection else torch.empty((0, xt.shape[1]))
    return xt, xa, xp


@torch.no_grad()
def build_updates(pipe, pairs, retains, args):
    # {group_name: {"parent": module, "weights": {param_name: tensor}}}
    groups = candidate_groups(pipe.unet, args.params)
    cache = TextCache(pipe)
    transforms, original, updated, diagnostics = {}, {}, {}, {}
    for group, entry in groups.items():
        parent = entry["parent"]
        # Standard SD1.x layers consume exactly the same text inputs. Reuse the
        # solve only when there is no per-attention normalization.
        input_key = group if getattr(parent, "norm_cross", False) else "plain_text"
        if input_key not in transforms:
            xt, xa, xp = module_inputs(
                parent, cache, pairs, retains, args.token_mode,
                args.include_padding, args.protect_null,
            )
            transforms[input_key] = solve_transform(
                xt, xa, xp, 
                beta=args.beta, 
                rank_rtol=args.rank_rtol,
                preserve_rank=args.preserve_rank, device=args.solve_device,
            )
        transform, info = transforms[input_key]
        diagnostics[group] = dict(info)
        for key, linear in entry["weights"].items():
            w = linear.weight.detach().cpu().clone()
            # Match the dtype in which sampling and checkpoint reload run.
            edited = (w.double() + w.double() @ transform).to(w.dtype)
            if not torch.isfinite(edited).all():
                raise ValueError(f"Non-finite edited weight: {key}")
            original[key] = w
            updated[key] = edited
        print(f"[solve] {group}: retain rank={info['protected_rank']}, null dim={info['null_dim']}", flush=True)
    group_keys = {name: list(entry["weights"]) for name, entry in groups.items()}
    return EditController(pipe.unet, original, updated, group_keys), diagnostics


class EditController:
    """Apply absolute original/edited values; never accumulate patches."""

    def __init__(self, unet, original, edited, groups, architecture="sd1"):
        self.parameters = dict(unet.named_parameters())
        self.original = original
        self.edited = edited
        self.groups = groups
        self.architecture = architecture
        self.active: set[str] = set()
        for key, value in edited.items():
            if key not in self.parameters or self.parameters[key].shape != value.shape:
                raise ValueError(f"Checkpoint does not match U-Net parameter: {key}")

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


def load_controller(unet, directory: Path, architecture="sd1"):
    directory = directory.parent if directory.is_file() else directory
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != f"casu2-{architecture}-v1":
        raise ValueError(f"Not a CASU-2 {architecture} checkpoint.")
    weights = load_file(str(directory / "weights.safetensors"), device="cpu")
    params = dict(unet.named_parameters())
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
    return EditController(unet, originals, weights, groups, architecture), metadata
