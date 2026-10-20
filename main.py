"""Unified command line for SD1.x, SDXL, SD3.0 and FLUX.2/FLUX.1."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import torch

from models import DEFAULT_MODELS, architecture_from_metadata, is_klein, resolve_settings
from models import backbone, build_updates, load_pipeline
from sampling import render_comparison, sample_records
from prompts import build_pairs, load_retain_texts, pair_records, split_concepts
from metrics import metrics_main

from editing import load_controller


ROOT = Path(__file__).resolve().parent


def add_model_args(parser, *, model_default=None):
    parser.add_argument("--architecture", choices=list(DEFAULT_MODELS), default=None)
    parser.add_argument("--sd-ckpt", "--sd_ckpt", default=model_default, help="HF model ID or local Diffusers directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--model-cpu-offload", action=argparse.BooleanOptionalAction, default=None,
                        help="Model CPU offload with CUDA; default on for FLUX.2-klein, off otherwise.")
    parser.add_argument("--max-sequence-length", type=int, default=None, help="Text length: Klein 512, T5 256 by default.")
    parser.add_argument("--flux-variant", choices=["klein", "dev", "schnell"], default=None,
                        help="Read from checkpoint when sampling; new FLUX models default to klein.")


def add_sampling_args(parser):
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0, help="FLUX true CFG, separate from embedded guidance.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None, help="Samples per prompt; benchmark 10, otherwise 1.")
    parser.add_argument("--sample-template-count", type=int, default=0, help="0 means all task templates.")
    parser.add_argument("--sample-template-offset", type=int, default=0)
    parser.add_argument("--contents", default=None, help="Concepts/artists to visualize; default: erasure targets.")
    parser.add_argument("--modes", default="original,edit")

# 验证采样用的参数
def validate_sampling(args):
    if min(args.steps, args.batch_size, args.num_samples, args.cpu_threads) <= 0:
        raise ValueError("Steps, batch size, samples and CPU threads must be positive.")
    if min(args.height, args.width) < 8 or args.height % 8 or args.width % 8:
        raise ValueError("Image dimensions must be positive multiples of 8.")
    if args.guidance_scale < 0 or args.seed < 0:
        raise ValueError("Guidance and seed must be nonnegative.")
    if args.max_sequence_length <= 0 or args.max_sequence_length > 512:
        raise ValueError("Text maximum sequence length must be in [1, 512].")
    if args.architecture in {"sd3", "flux"} and (args.height % 16 or args.width % 16):
        raise ValueError("SD3/FLUX image dimensions must be multiples of 16.")
    if args.architecture == "flux":
        if is_klein(args) and (args.guidance_scale != 1 or args.true_cfg_scale != 1 or args.negative_prompt):
            raise ValueError("Distilled FLUX.2-klein-4B uses guidance-scale=1 without negative prompts or true CFG.")
        if args.flux_variant == "schnell" and (args.guidance_scale != 0 or args.max_sequence_length > 256):
            raise ValueError("FLUX.1-schnell requires guidance-scale=0 and max-sequence-length<=256.")
        if args.true_cfg_scale < 1 or (args.negative_prompt and args.true_cfg_scale <= 1):
            raise ValueError("FLUX negative prompts require --true-cfg-scale > 1; the default is no true CFG.")
    modes = args.modes.replace(" ", "").split(",")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"original", "edit"}:
        raise ValueError("--modes must contain original, edit, or original,edit.")
    args.modes = modes


def new_output(path, prefix, overwrite=False):
    if path:
        directory = Path(path).expanduser().resolve()
    else:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y_%m_%d_%H_%M_%S")
        directory = ROOT / "runs" / f"{prefix}-{stamp}"
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is nonempty: {directory}. Choose a new path or --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def edit_parser():
    parser = argparse.ArgumentParser(prog="main.py erase", description="CASU-2 closed-form concept erasure and image generation.")
    add_model_args(parser)
    add_sampling_args(parser)
    parser.add_argument("--task", "--erasetype", choices=["instance", "style"], required=True)
    parser.add_argument("--target-concepts", "--target_concepts", required=True)
    parser.add_argument("--anchor-concepts", "--anchor_concepts", required=True, help="One anchor, one per target, or an empty string.")
    parser.add_argument("--contextual-anchor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context-template-count", type=int, default=30, help="0 means all task templates.")
    # token option
    parser.add_argument("--token-mode", choices=["full", "subject"], default="full")
    parser.add_argument("--include-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-branches", choices=["both", "tokens", "pooled"], default="both")
    parser.add_argument("--max-fit-tokens", type=int, default=16, help="SDXL/SD3/FLUX rows per segment/pair; 0 keeps all.")
    parser.add_argument("--encode-batch-size", type=int, default=None, help="Klein defaults to 1; other architectures to 4.")
    # hyperparameter
    parser.add_argument("--params", choices=["V", "K", "KV"], default="V")
    # beta 限制权重更新幅度
    parser.add_argument("--beta", type=float, default=1.0)
    # 只保留retain矩阵中，奇异值比 最大奇异值的rank_rtol倍大奇异值对应的向量，其他向量被丢弃
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    # preserve_rank: 如果不为None，则在计算retain矩阵的奇异值分解时，保留的奇异值个数不能超过preserve_rank
    parser.add_argument("--preserve-rank", type=int, default=None, help="Optional explicit rank truncation; relaxes full retain protection.")
    # path
    parser.add_argument("--protect-null", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retain-path", type=Path, default=None, help="Default: data/instance.csv or data/style.csv.")
    parser.add_argument("--retain-column", default="concept")
    parser.add_argument("--retain-count", type=int, default=128, help="0 disables external retains; -1 uses all (may exhaust null space).")
    parser.add_argument("--solve-device", default=None, help="Closed-form solver device; defaults to --device.")
    # selective update
    parser.add_argument("--selective-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selection-method", "--sd1-su-method", dest="selection_method",
                        choices=["middle", "vlm", "legacy"], default=None,
                        help="Shared SU: middle U-Net layers, staged VLM search, or legacy deletion. Default: SD1 middle, others legacy.")
    parser.add_argument("--su-min-fraction", type=float, default=0.5)
    parser.add_argument("--su-max-fraction", type=float, default=0.75,
                        help="Upper candidate fraction. A one-entry adapter requires explicit 1 for VLM SU.")
    parser.add_argument("--su-prompts", type=Path, help="VLM calibration JSON; default: data/su_prompts.json.")
    parser.add_argument("--su-vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Local Transformers vision-language model, loaded after SD is unloaded.")
    parser.add_argument("--su-vlm-image-size", type=int, default=448)
    parser.add_argument("--selection-order", choices=["proxy", "forward", "reverse", "random"], default="proxy")
    parser.add_argument("--selection-judge", choices=["manual", "vlm"], default="manual")
    parser.add_argument("--selection-template-count", type=int, default=3)
    parser.add_argument("--selection-samples", type=int, default=1)
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--probe-points", type=int, default=3)
    parser.add_argument("--probe-retains", type=int, default=2)
    
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-key-env", default="VLM_API_KEY")
    parser.add_argument("--judge-timeout", type=float, default=120)
    parser.add_argument("--target-description", default="", help="Optional visual concept/style definition for the VLM.")
    
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-path-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def edit_main(argv=None):
    parser = edit_parser()
    args = parser.parse_args(argv)
    try:
        resolve_settings(args)
        # Retain old defaults while sharing every selection workflow across models.
        args.selection_method = args.selection_method or ("middle" if args.architecture == "sd1" else "legacy")
        args.solve_device = args.solve_device or args.device
        validate_sampling(args)
        args.targets = split_concepts(args.target_concepts)
        anchors = split_concepts(args.anchor_concepts, allow_empty=True)

        if args.beta <= 0 or args.rank_rtol < 0 or args.retain_count < -1:
            raise ValueError("Invalid ridge, rank tolerance or retain count.")
        if args.preserve_rank is not None and args.preserve_rank < 0:
            raise ValueError("--preserve-rank must be nonnegative.")
        if args.max_fit_tokens < 0 or args.encode_batch_size <= 0:
            raise ValueError("max-fit-tokens must be nonnegative and encode-batch-size positive.")
        if args.architecture == "sd1" and args.condition_branches == "pooled":
            raise ValueError("SD1.x has no separate pooled-text injection branch.")
        if is_klein(args) and args.condition_branches == "pooled":
            raise ValueError("FLUX.2-klein has no pooled-text branch; use tokens or both (sequence entry only).")
        if min(args.selection_template_count, args.selection_samples, args.probe_steps, args.probe_points) <= 0:
            raise ValueError("Selection/probe counts must be positive.")
        if args.probe_points > args.probe_steps or args.probe_retains < 0 or args.judge_timeout <= 0:
            raise ValueError("Invalid probe counts or judge timeout.")
        staged_su = args.selective_update and args.selection_method != "legacy"
        if not 0 < args.su_min_fraction <= args.su_max_fraction <= 1:
            raise ValueError("SU fractions must satisfy 0 < --su-min-fraction <= --su-max-fraction <= 1.")
        if args.su_vlm_image_size < 56:
            raise ValueError("--su-vlm-image-size must be at least 56.")
        if args.selective_update and not staged_su and args.selection_judge == "vlm" and not (args.judge_base_url and args.judge_model):
            raise ValueError("VLM selection needs --judge-base-url and --judge-model.")
        
        pairs = build_pairs(
            args.targets, anchors, args.task, args.contextual_anchor, args.context_template_count)
        if args.retain_path is None:
            args.retain_path = ROOT / "data" / f"{args.task}.csv"
        else:
            args.retain_path = args.retain_path.expanduser().resolve()
        all_retains = load_retain_texts(
            args.retain_path, args.retain_column, args.targets, -1,
        ) if args.retain_count != 0 else []
        retains = all_retains if args.retain_count < 0 else all_retains[:args.retain_count]
        
        # holdout_retains: selective update 排序使用的非目标探测文本
        retained_set = set(retains)
        heldout_retains = [x for x in all_retains if x not in retained_set][:args.probe_retains]
        if not heldout_retains:
            fallback = ["a photo of a river.", "a photo of a chair.", "a landscape of a mountain."]
            heldout_retains = [
                x for x in fallback if not any(t.casefold() in x.casefold() for t in args.targets)
            ][:args.probe_retains]
        # contents用于生成
        contents = split_concepts(args.contents) if args.contents else args.targets
        records = sample_records(
            contents, args.task, args.sample_template_count, args.num_samples,
            args.seed, args.sample_template_offset,
        )
        selection_records = sample_records(
            args.targets, args.task, args.selection_template_count, args.selection_samples, args.seed,
        )
        if staged_su and args.selection_method == "vlm":
            from selection import calibration_records
            selection_records = calibration_records(
                args.targets, args.task, args.selection_template_count,
                args.selection_samples, args.seed, args.su_prompts,
            )
        prefix = args.task if args.architecture == "sd1" else f"{args.architecture}-{args.task}"
        output = new_output(args.output_dir, prefix, args.overwrite)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    
    config = dict(vars(args))
    write_json(output / "config.json", config)
    write_json(output / "prompt_pairs.json", pair_records(pairs))
    write_json(output / "retain_texts.json", retains)
    write_json(output / "probe_retain_texts.json", heldout_retains)
    write_json(output / "generation_prompts.json", records)
    print(f"[plan] pairs={len(pairs)}, retain concepts={len(retains)}, final images/mode={len(records)}")
    print(f"[output] {output}", flush=True)

    # edit begins here
    # torch.set_num_threads(args.cpu_threads)
    pipe = load_pipeline(args)
    controller, diagnostics = build_updates(pipe, pairs, retains, args)
    candidate_group_count = len(controller.groups)
    active = set(controller.groups)
    metadata = {
        "base_model": args.sd_ckpt, "task": args.task, "targets": args.targets,
        "architecture": args.architecture,
        "anchors": anchors, "config": config, "solver": diagnostics,
        "weight_dtype": str(next(backbone(pipe, args.architecture).parameters()).dtype).removeprefix("torch."),
    }
    selection = {"status": "disabled"}
    checkpoint = output / "checkpoint"
    checkpoint_ready = False
    if staged_su:
        from selection import selection_plan
        try:
            plan = selection_plan(controller.groups, args.selection_method, args.su_min_fraction, args.su_max_fraction)
        except ValueError as error:
            parser.error(str(error))
        selection_dir = output / "selection"
        selection_dir.mkdir(parents=True, exist_ok=True)
        write_json(selection_dir / "layer_plan.json", plan)
        if args.selection_method == "middle":
            active = set(plan["middle_groups"])
            selection = {
                "method": "middle", "status": "heuristic_no_judge", "unit_kind": plan["unit_kind"],
                "total_groups": plan["total_groups"], "selected_group_count": len(active),
                "total_layers": plan["total_layers"], "selected_layer_count": len(active),
                "active_groups": sorted(active),
            }
            print(f"[SU:middle] updating {len(active)}/{plan['total_layers']} layers; no VLM", flush=True)
        else:
            from selection import prepare_candidates, judge_candidates, publish_candidate, release_cuda
            # Explicit --overwrite must not leave an old successful checkpoint if calibration fails.
            if checkpoint.exists():
                import shutil
                shutil.rmtree(checkpoint)
            state = prepare_candidates(pipe, controller, plan, selection_records, selection_dir, args, metadata)
            # Both references are significant: controller.parameters also owns U-Net tensors.
            # Destroy all diffusion state before constructing the local VLM.
            if getattr(pipe, "_casu_cpu_offload", False):
                pipe.remove_all_hooks()
            del controller, pipe
            release_cuda()
            print("[SU:memory] diffusion unloaded; loading VLM", flush=True)
            chosen, selection = judge_candidates(state, selection_dir, args)
            release_cuda()
            if chosen is None:
                write_json(output / "summary.json", {
                    "checkpoint": None, "candidate_group_count": candidate_group_count,
                    "selection": selection, "quality_metrics_run": False,
                })
                print(f"[SU] No verified candidate: {selection['status']}; see {selection_dir / 'judgments.json'}", flush=True)
                return 2
            publish_candidate(selection_dir, chosen, checkpoint, selection)
            checkpoint_ready = True
            active = set(chosen["active_groups"])
            if args.generate:
                # The judge has already been closed and its CUDA memory freed.
                pipe = load_pipeline(args)
                controller, _ = load_controller(backbone(pipe, args.architecture), checkpoint, args.architecture)
    elif args.selective_update:
        controller.apply(active)
        # A complete checkpoint survives interrupted or unavailable judging.
        controller.save(output / "checkpoint_all", {**metadata, "selection": {"status": "all_updates_unverified"}})
        from selection import run_selection
        active, selection = run_selection(
            pipe, controller, pairs, heldout_retains, selection_records, output / "selection", args,
        )
    
    if not checkpoint_ready:
        controller.apply(active)
        controller.save(checkpoint, {**metadata, "selection": selection})
    if args.checkpoint_path_file:
        args.checkpoint_path_file.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_path_file.write_text(str(checkpoint), encoding="utf-8")
    print(f"[checkpoint] {checkpoint}", flush=True)
    # 生成对比html
    if args.generate:
        render_comparison(pipe, controller, active, records, output / "images", args, args.modes)
        print(f"[gallery] {output / 'images' / 'index.html'}", flush=True)
    write_json(output / "summary.json", {
        "checkpoint": str(checkpoint), "active_groups": sorted(active),
        "candidate_group_count": candidate_group_count, "selection": selection,
        "image_count_per_mode": len(records) if args.generate else 0,
        "quality_metrics_run": False,
    })
    
    if args.selective_update and not staged_su and (
        not selection["initial_verified_clear"]
        or selection["status"] in {"judge_inconclusive", "all_updates_removed_check_judge"}
    ):
        print(f"[selection] Unverified outcome: {selection['status']}. Checkpoints/images were kept.")
        return 2
    return 0


def sample_main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py sample", description="Generate matched original/edited images from a CASU-2 checkpoint.")
    add_model_args(parser)
    add_sampling_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Paper's full five-character or three-artist evaluation set.")
    parser.add_argument("--prompt-csv", type=Path, help="Optional full dataset CSV with concept,prompt columns.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching valid PNGs; reject different checkpoint/prompts/settings.")
    parser.add_argument("--original-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse original images in task/concept folders (default: on).")
    parser.add_argument("--original-dir", "--original-cache-dir", dest="original_dir", type=Path,
                        help="Original root; defaults under runs/: SD1 original, SDXL original_SDXL, SD3 original_SD3, FLUX original_FLUX.")
    parser.add_argument("--save-comparisons", action=argparse.BooleanOptionalAction, default=None,
                        help="Save side-by-side composites; off for benchmark, on otherwise.")
    args = parser.parse_args(argv)
    try:
        # sample 时会加载checkpoint的metadata.json，里面包含了architecture和base_model等信息，必须与命令行参数一致
        checkpoint = args.checkpoint.expanduser().resolve()
        checkpoint = checkpoint.parent if checkpoint.is_file() else checkpoint
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        architecture = architecture_from_metadata(metadata)
        if args.architecture and args.architecture != architecture:
            raise ValueError("--architecture does not match the checkpoint.")
        if args.sd_ckpt is not None and args.sd_ckpt != metadata["base_model"]:
            raise ValueError("--sd-ckpt must match the base model recorded in the checkpoint.")
        args.sd_ckpt = metadata["base_model"]
        if args.flux_variant is None:
            args.flux_variant = metadata.get("config", {}).get("flux_variant")
        if args.max_sequence_length is None:
            args.max_sequence_length = metadata.get("config", {}).get("max_sequence_length", 256)
        for name in ("height", "width", "steps", "guidance_scale"):
            if getattr(args, name) is None:
                setattr(args, name, metadata.get("config", {}).get(name))
        if args.num_samples is None:
            args.num_samples = 10 if args.benchmark else 1
        if args.prompt_csv:
            args.prompt_csv = args.prompt_csv.expanduser().resolve()
        if args.save_comparisons is None:
            args.save_comparisons = not args.benchmark
        resolve_settings(args, architecture)
        if args.original_dir is None:
            args.original_dir = ROOT / "runs" / {
                "sd1": "original", "sdxl": "original_SDXL",
                "sd3": "original_SD3", "flux": "original_FLUX",
            }[args.architecture]
        validate_sampling(args)
        from evaluation import dataset_records, prepare_sample_plan
        records = dataset_records(args, metadata)
        if args.resume and args.output_dir is None:
            raise ValueError("--resume requires an explicit --output-dir.")
        output = new_output(args.output_dir, "benchmark" if args.benchmark else "sample", args.overwrite or args.resume)
        prepare_sample_plan(output, checkpoint, metadata, records, args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    write_json(output / "config.json", vars(args))
    write_json(output / "generation_prompts.json", records)
    print(f"[dataset] {len(records)} images/mode; output={output}", flush=True)


    # torch.set_num_threads(args.cpu_threads)
    pipe = load_pipeline(args)
    controller, metadata = load_controller(backbone(pipe, args.architecture), checkpoint, args.architecture)
    active = set(metadata["active_groups"])
    originals = None
    if args.original_cache and "original" in args.modes:
        from originals import OriginalImages, original_config
        originals = OriginalImages(args.original_dir, metadata["task"], original_config(pipe, args))
    render_comparison(pipe, controller, active, records, output, args, args.modes,
                      resume=args.resume, comparisons=args.save_comparisons, originals=originals)
    print(f"[gallery] {output / 'index.html'}", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"erase": edit_main, "sample": sample_main, "metrics": metrics_main}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    
    parser = argparse.ArgumentParser(
        description="CASU-2 SD1.x/SDXL/SD3/FLUX: erase concepts or sample a saved checkpoint.",
    )
    parser.add_argument("command", choices=commands, help="Use COMMAND --help for its options.")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
