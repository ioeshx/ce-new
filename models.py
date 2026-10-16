"""Architecture selection, pipeline loading and checkpoint compatibility."""

from __future__ import annotations

import importlib
import torch

DEFAULT_MODELS = {
    "sd1": "CompVis/stable-diffusion-v1-4",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.2-klein-4B",
}

FLUX_MODELS = {
    "klein": DEFAULT_MODELS["flux"],
    "dev": "black-forest-labs/FLUX.1-dev",
    "schnell": "black-forest-labs/FLUX.1-schnell",
}


def is_klein(args):
    return args.architecture == "flux" and args.flux_variant == "klein"


def architecture_from_metadata(metadata):
    architecture = metadata.get("architecture")
    if architecture is None:
        architecture = next(
            (name for name in DEFAULT_MODELS if metadata.get("format") == f"casu2-{name}-v1"),
            None,
        )
    if architecture not in DEFAULT_MODELS or metadata.get("format") != f"casu2-{architecture}-v1":
        raise ValueError("Not a supported CASU-2 checkpoint.")
    return architecture


def resolve_settings(args, architecture=None):
    args.architecture = architecture or args.architecture or "sd1"
    if args.flux_variant is None:
        model_name = (args.sd_ckpt or "").lower()
        args.flux_variant = (
            "schnell" if "flux.1-schnell" in model_name
            else "dev" if "flux.1-dev" in model_name
            else "klein" if args.architecture == "flux" else "dev"
        )
    klein = is_klein(args)
    if args.max_sequence_length is None:
        args.max_sequence_length = 512 if klein else 256
    if args.sd_ckpt is None:
        args.sd_ckpt = DEFAULT_MODELS[args.architecture]
        if args.architecture == "flux":
            args.sd_ckpt = FLUX_MODELS[args.flux_variant]
    fast_flux = args.architecture == "flux" and args.flux_variant in {"klein", "schnell"}
    if args.steps is None:
        args.steps = 4 if fast_flux else 28 if args.architecture in {"sd3", "flux"} else 20
    if args.guidance_scale is None:
        args.guidance_scale = 1.0 if klein else 0.0 if fast_flux else {
            "sd1": 7.5, "sdxl": 5.0, "sd3": 7.0, "flux": 3.5,
        }[args.architecture]
    if args.height is None:
        args.height = 512 if args.architecture == "sd1" else 1024
    if args.width is None:
        args.width = 512 if args.architecture == "sd1" else 1024
    if args.model_cpu_offload is None:
        args.model_cpu_offload = klein and args.device.startswith("cuda")
    if hasattr(args, "encode_batch_size") and args.encode_batch_size is None:
        args.encode_batch_size = 1 if klein else 4
    if hasattr(args, "num_samples") and args.num_samples is None:
        args.num_samples = 1


def backbone(pipe, architecture):
    return pipe.unet if architecture in {"sd1", "sdxl"} else pipe.transformer


def pipeline_device(pipe):
    return pipe._execution_device


def load_pipeline(args):
    if args.architecture == "sd1":
        if args.model_cpu_offload:
            raise ValueError("--model-cpu-offload is supported for SDXL/SD3/FLUX; SD1 keeps its original loader.")
        from sd import load_pipeline as load_sd1
        pipe = load_sd1(args.sd_ckpt, args.device, args.dtype, args.local_files_only)
    else:
        from diffusers import (
            DPMSolverMultistepScheduler, FluxPipeline, StableDiffusion3Pipeline,
            StableDiffusionXLPipeline,
        )
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use a GPU environment or explicitly --device cpu.")
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
        classes = {"sdxl": StableDiffusionXLPipeline, "sd3": StableDiffusion3Pipeline, "flux": FluxPipeline}
        if is_klein(args):
            try:
                from diffusers import Flux2KleinPipeline
            except ImportError as error:
                raise RuntimeError(
                    "FLUX.2-klein requires Diffusers >=0.37.1. Install casu-2/requirements.txt."
                ) from error
            classes["flux"] = Flux2KleinPipeline
        kwargs = dict(torch_dtype=getattr(torch, dtype), local_files_only=args.local_files_only)
        if args.architecture == "sdxl":
            kwargs["add_watermarker"] = False
        pipe = classes[args.architecture].from_pretrained(args.sd_ckpt, **kwargs)
        if args.architecture == "sdxl":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if args.architecture == "sd3" and pipe.text_encoder_3 is None:
            raise ValueError("SD3 adapter requires the T5 encoder; omitting it changes the conditioning problem.")
        if args.architecture == "flux":
            actual_guidance = bool(pipe.transformer.config.guidance_embeds)
            if is_klein(args):
                if pipe.text_encoder.config.model_type != "qwen3" or not pipe.config.is_distilled or actual_guidance:
                    raise ValueError("The klein adapter expects the distilled FLUX.2-klein-4B Qwen3 pipeline.")
                if pipe.transformer.config.joint_attention_dim != pipe.text_encoder.config.hidden_size * 3:
                    raise ValueError("Klein context width must match the concatenated three Qwen3 hidden layers.")
            elif actual_guidance != (args.flux_variant == "dev"):
                raise ValueError("FLUX.1 variant does not match transformer.guidance_embeds; set --flux-variant dev/schnell.")
        if args.model_cpu_offload:
            pipe.enable_model_cpu_offload(device=args.device)
        else:
            pipe.to(args.device)
        pipe._casu_cpu_offload = args.model_cpu_offload
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_slicing()
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "unet", "transformer", "vae"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.eval()
    pipe._casu_architecture = args.architecture
    return pipe


def build_updates(pipe, pairs, retains, args):
    adapter = importlib.import_module("sd" if args.architecture == "sd1" else args.architecture)
    return adapter.build_updates(pipe, pairs, retains, args)
