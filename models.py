"""Architecture selection, pipeline loading and checkpoint compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sd import build_updates as sd1_build_updates
from sdxl import build_updates as sdxl_build_updates
from sd3 import build_updates as sd3_build_updates
from flux import build_updates as flux_build_updates

# FLUX.2-klein requires Diffusers >=0.37.1.
from diffusers import (
    DPMSolverMultistepScheduler, 
    Flux2KleinPipeline,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
)

# Model card for diffusers
DEFAULT_MODELS = {
    "sd1": "CompVis/stable-diffusion-v1-4",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.2-klein-4B",
}

def architecture_from_metadata(metadata):
    architecture = metadata.get("architecture")
    if architecture is None:
        architecture = next(
            (name for name in DEFAULT_MODELS if metadata.get("format") == f"casu2-{name}-v1"),
            None,
        )
    if architecture not in DEFAULT_MODELS or metadata.get("format") != f"casu2-{architecture}-v1":
        raise ValueError("Not a supported CASU-2 checkpoint.")
    if architecture == "flux" and metadata.get("config", {}).get("flux_variant", "klein") not in {None, "klein"}:
        raise ValueError("FLUX checkpoints must use black-forest-labs/FLUX.2-klein-4B.")
    return architecture


def validate_flux_source(model):
    """Accept the supported Hub model or a local Diffusers copy of Klein."""
    if model == DEFAULT_MODELS["flux"]:
        return
    directory = Path(model).expanduser()
    index = directory / "model_index.json"
    if not index.is_file() or json.loads(index.read_text(encoding="utf-8")).get("_class_name") != "Flux2KleinPipeline":
        raise ValueError(
            "FLUX supports only black-forest-labs/FLUX.2-klein-4B or a local Diffusers copy using Flux2KleinPipeline."
        )


# 根据对应model设置参数
# steps = 20(sd1,sdxl), 28(sd3), 4(flux)
# guidance_scale = 7.5(sd1), 5.0(sdxl), 7.0(sd3), 1.0(flux)
def resolve_settings(args, architecture=None):
    args.architecture = architecture or args.architecture or "sd1"
    flux = args.architecture == "flux"
    # max seq len, sd ckpt
    # steps, guidance_scale, height, width, model_cpu_offload, encode_batch_size, num_samples
    if args.max_sequence_length is None:
        args.max_sequence_length = 512 if flux else 256
    if args.sd_ckpt is None:
        args.sd_ckpt = DEFAULT_MODELS[args.architecture]
    if args.steps is None:
        args.steps = {"sd1": 20, "sdxl": 20, "sd3": 28, "flux": 4}[args.architecture]
    if args.guidance_scale is None:
        args.guidance_scale = {
            "sd1": 7.5, "sdxl": 5.0, "sd3": 7.0, "flux": 1.0,
        }[args.architecture]
    if args.height is None:
        args.height = 512 if args.architecture == "sd1" else 1024
    if args.width is None:
        args.width = 512 if args.architecture == "sd1" else 1024
    if args.model_cpu_offload is None:
        args.model_cpu_offload = flux and args.device.startswith("cuda")
    if hasattr(args, "encode_batch_size") and args.encode_batch_size is None:
        args.encode_batch_size = 1 if flux else 4
    if hasattr(args, "num_samples") and args.num_samples is None:
        args.num_samples = 1

def backbone(pipe, architecture):
    return pipe.unet if architecture in {"sd1", "sdxl"} else pipe.transformer

def pipeline_device(pipe):
    return pipe._execution_device

# dtype:
# SD,SDXL: float16
# SD3,flux: bfloat16
def load_pipeline(args):
    if args.architecture == "flux":
        validate_flux_source(args.sd_ckpt)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU environment or explicitly --device cpu.")
    if args.architecture == "sd1" and args.model_cpu_offload:
        raise ValueError("--model-cpu-offload is supported for SDXL/SD3/FLUX; SD1 runs on --device.")
    if args.model_cpu_offload and not args.device.startswith("cuda"):
        raise ValueError("Model CPU offload requires a CUDA execution device.")

    dtype = args.dtype
    if dtype == "auto":
        if not args.device.startswith("cuda"):
            dtype = "float32"
        elif args.architecture in {"sd3", "flux"} and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float16"
    if args.device == "cpu" and dtype == "float16":
        raise ValueError("Use float32 for CPU execution.")

    classes = {
        "sd1": StableDiffusionPipeline,
        "sdxl": StableDiffusionXLPipeline,
        "sd3": StableDiffusion3Pipeline,
        "flux": Flux2KleinPipeline,
    }
    kwargs = dict(torch_dtype=getattr(torch, dtype), local_files_only=args.local_files_only)
    if args.architecture == "sd1":
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    elif args.architecture == "sdxl":
        kwargs["add_watermarker"] = False
    pipe = classes[args.architecture].from_pretrained(args.sd_ckpt, **kwargs)

    if args.architecture in {"sd1", "sdxl"}:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.architecture == "sd1":
        cross_dim = pipe.unet.config.cross_attention_dim
        cross_dims = cross_dim if isinstance(cross_dim, (tuple, list)) else [cross_dim]
        if int(pipe.text_encoder.config.hidden_size) != 768 or set(cross_dims) != {768}:
            raise ValueError("This implementation targets SD1.x (768-D CLIP). SDXL/SD2/SD3/FLUX need adapters.")
    if args.architecture == "sd3" and pipe.text_encoder_3 is None:
        raise ValueError("SD3 adapter requires the T5 encoder; omitting it changes the conditioning problem.")
    if args.architecture == "flux":
        actual_guidance = bool(pipe.transformer.config.guidance_embeds)
        if pipe.text_encoder.config.model_type != "qwen3" or not pipe.config.is_distilled or actual_guidance:
            raise ValueError("The flux adapter expects the distilled FLUX.2-klein-4B Qwen3 pipeline.")
        if pipe.transformer.config.joint_attention_dim != pipe.text_encoder.config.hidden_size * 3:
            raise ValueError("Klein context width must match the concatenated three Qwen3 hidden layers.")

    if args.model_cpu_offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    pipe._casu_cpu_offload = args.model_cpu_offload
    pipe._casu_architecture = args.architecture
    pipe.set_progress_bar_config(disable=True)
    if args.architecture == "sd1":
        pipe.enable_attention_slicing()
    else:
        pipe.vae.enable_slicing()
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "unet", "transformer", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.eval()
    return pipe


def build_updates(pipe, pairs, retains, args):
    builders = {
        "sd1": sd1_build_updates,
        "sdxl": sdxl_build_updates,
        "sd3": sd3_build_updates,
        "flux": flux_build_updates,
    }
    return builders[args.architecture](pipe, pairs, retains, args)
