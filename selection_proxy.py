"""Native-pipeline trajectory probes for SDXL, SD3 and FLUX selection."""

from __future__ import annotations

from collections import defaultdict
import copy
import torch

from models import backbone, is_klein, pipeline_device
from sampling import pipeline_arguments


def tree_map(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {k: tree_map(v, function) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(v, function) for v in value)
    if isinstance(value, list):
        return [tree_map(v, function) for v in value]
    return value


def prediction(output):
    if torch.is_tensor(output):
        return output
    return output.sample if hasattr(output, "sample") else output[0]


class _CapturedInputs(Exception):
    pass


@torch.inference_mode()
def capture(pipe, prompt, args, *, inputs_only=False, seed=None):
    """Record real backbone inputs; use the native scheduler/packing/guidance.

    inputs_only aborts at the first backbone pre-hook after prompt encoding.
    No VAE decode or image classifier is involved.
    """
    options = copy.copy(args)
    options.steps = args.probe_steps
    record = {"prompt": prompt, "seed": args.seed if seed is None else seed}
    model = backbone(pipe, args.architecture)
    selected = set(torch.linspace(0, args.probe_steps - 1, args.probe_points).round().int().tolist())
    shots, pending = [], {}
    calls = 0
    double_calls = args.architecture == "flux" and not is_klein(args) and args.true_cfg_scale > 1

    def before(module, positional, keywords):
        nonlocal calls
        step, positive = (calls // 2, calls % 2 == 0) if double_calls else (calls, True)
        calls += 1
        pending.clear()
        if inputs_only or (positive and step in selected):
            snapshot = {
                "positional": tree_map(positional, lambda t: t.detach().cpu().clone()),
                "keywords": tree_map(keywords, lambda t: t.detach().cpu().clone()),
            }
            if inputs_only:
                shots.append(snapshot)
                raise _CapturedInputs()
            pending.update(snapshot)

    def after(module, positional, keywords, output):
        if pending:
            shots.append({**pending, "original": prediction(output).detach().float().cpu()})
            pending.clear()

    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    try:
        pipe(**pipeline_arguments(pipe, [record], options), output_type="latent")
    except _CapturedInputs:
        if not inputs_only:
            raise
    finally:
        pre.remove()
        post.remove()
        if getattr(pipe, "_casu_cpu_offload", False):
            pipe.maybe_free_model_hooks()
    if not shots:
        raise ValueError("No backbone states captured by the native pipeline.")
    return shots


def reference_keywords(target, reference):
    result = dict(target)
    for name in ("encoder_hidden_states", "pooled_projections", "txt_ids"):
        if name in reference:
            result[name] = reference[name]
    if "added_cond_kwargs" in target:
        result["added_cond_kwargs"] = dict(target["added_cond_kwargs"])
        result["added_cond_kwargs"]["text_embeds"] = reference["added_cond_kwargs"]["text_embeds"]
    return result


@torch.inference_mode()
def forward_shot(pipe, shot, keywords=None):
    device = pipeline_device(pipe)
    positional = tree_map(shot["positional"], lambda t: t.to(device))
    keywords = tree_map(shot["keywords"] if keywords is None else keywords, lambda t: t.to(device))
    return prediction(backbone(pipe, pipe._casu_architecture)(*positional, **keywords)).detach().float().cpu()


@torch.inference_mode()
def proxy_order_generic(pipe, controller, pairs, retains, args):
    chosen = {}
    for pair in pairs:
        chosen.setdefault(pair.concept, pair)
    specs = [(p.target_prompt, p.anchor_prompt, p.concept) for p in chosen.values()]
    specs += [(text, None, None) for text in retains[:args.probe_retains]]
    if not any(c is None for _, _, c in specs):
        specs.append(("a landscape of a mountain.", None, None))
    probes = []
    full = set(controller.groups)
    controller.apply(set())
    try:
        for index, (prompt, reference, concept) in enumerate(specs):
            shots = capture(pipe, prompt, args, seed=args.seed + index)
            safe = capture(pipe, reference, args, inputs_only=True)[0] if reference is not None else None
            for shot in shots:
                output = forward_shot(
                    pipe, shot, reference_keywords(shot["keywords"], safe["keywords"]),
                ) if safe is not None else shot["original"]
                denominator = float((shot["original"] - output).square().mean())
                if concept is None or denominator > 1e-10:
                    probes.append({
                        "shot": shot, "concept": concept, "reference": output,
                        "denom": max(denominator, 1e-10),
                    })

        def evaluate(active):
            controller.apply(active)
            residual, keep = defaultdict(list), []
            for probe in probes:
                error = float((forward_shot(pipe, probe["shot"]) - probe["reference"]).square().mean())
                if probe["concept"] is None:
                    keep.append(error)
                else:
                    residual[probe["concept"]].append(error / probe["denom"])
            if set(residual) != set(chosen):
                raise ValueError("At least one concept has no usable target-anchor prediction contrast.")
            return {c: sum(v) / len(v) for c, v in residual.items()}, sum(keep) / max(1, len(keep))

        baseline, damage = evaluate(full)
        scores = []
        for index, group in enumerate(controller.groups, 1):
            print(f"[selection proxy:{args.architecture}] {index}/{len(full)}: {group}", flush=True)
            residual, new_damage = evaluate(full - {group})
            scores.append({
                "group": group, "risk": max(residual[c] - baseline[c] for c in baseline),
                "preservation_gain": damage - new_damage,
                "parameter_count": sum(controller.edited[k].numel() for k in controller.groups[group]),
            })
        scores.sort(key=lambda r: (r["risk"], -r["preservation_gain"], -r["parameter_count"], r["group"]))
        return [r["group"] for r in scores], scores
    finally:
        controller.apply(full)
