"""Shared selective update: middle layers, staged VLM search and legacy deletion.

Architecture adapters supply edit groups. Generation uses each model's native
pipeline; generation and local VLM judging run in separate phases.
"""

from __future__ import annotations

import base64
import gc
import math
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
import urllib.error
import urllib.request

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from sampling import render_selection
from conditioning import ConditionCache


class JudgeUnavailable(RuntimeError):
    pass


def parse_judgment(content: str, targets: list[str]) -> tuple[str, dict]:
    """Malformed/incomplete responses can never be interpreted as clearance."""
    if not isinstance(content, str):
        raise ValueError("Judge content must be a JSON string.")
    content = content.strip()
    if content.startswith("~~~") or content.startswith(chr(96) * 3):
        content = "\n".join(content.splitlines()[1:-1])
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object.")
    concepts = value.get("concepts")
    if not isinstance(concepts, dict) or set(concepts) != set(targets):
        raise ValueError("Judge must return exactly the requested concept names.")
    states = []
    for target in targets:
        item = concepts[target]
        if not isinstance(item, dict) or "present" not in item:
            raise ValueError("Every target must explicitly include a present field.")
        state = item["present"]
        if state is not True and state is not False and state is not None:
            raise ValueError("present must be JSON true, false, or null.")
        states.append(state)
    status = "present" if any(x is True for x in states) else (
        "unknown" if any(x is None for x in states) else "clear"
    )
    return status, value


class ManualJudge:
    def evaluate(self, directory, records, targets):
        print(f"\n[selection] Open gallery: {directory / 'index.html'}", flush=True)
        print(f"Inspect ALL images for ANY of: {', '.join(targets)}", flush=True)
        while True:
            try:
                answer = input("Target present? [yes / no / unknown]: ").strip().casefold()
            except (EOFError, KeyboardInterrupt) as error:
                raise JudgeUnavailable("Manual judgment was not supplied; candidate is unverified.") from error
            if answer in {"yes", "y", "present", "1"}:
                return "present", {"judge": "manual", "answer": answer}
            if answer in {"no", "n", "clear", "0"}:
                return "clear", {"judge": "manual", "answer": answer}
            if answer in {"unknown", "u", "?"}:
                return "unknown", {"judge": "manual", "answer": answer}


class VisionJudge:
    """Optional OpenAI-compatible endpoint, including local VLM servers.

    No request occurs unless --selection-judge vlm is explicitly selected.
    Credentials come only from the configured environment variable.
    """

    def __init__(self, base_url, model, key_env, timeout, target_description=""):
        if not base_url or not model:
            raise ValueError("--judge-base-url and --judge-model are required for the VLM judge.")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model, self.key_env, self.timeout = model, key_env, timeout
        self.description = target_description

    def evaluate(self, directory, records, targets):
        responses, overall = [], "clear"
        schema = {"concepts": {t: {"present": None, "evidence": "brief visual evidence"} for t in targets}}
        instruction = (
            "Inspect the image, not a generation prompt. For EACH named concept, decide "
            "whether it is visually present. For artist names assess the visual art style, "
            "not a face or written artist name. Use null if uncertain. Return ONLY JSON "
            "with the exact concept keys and schema: " + json.dumps(schema)
            + "\nAdditional concept definitions: " + self.description
        )
        for row in records:
            encoded = base64.b64encode((directory / row["edit"]).read_bytes()).decode("ascii")
            payload = {
                "model": self.model, "temperature": 0, "max_tokens": 512,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
                ]}],
            }
            headers = {"Content-Type": "application/json"}
            key = os.environ.get(self.key_env)
            if key:
                headers["Authorization"] = "Bearer " + key
            try:
                request = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode("utf-8"), headers=headers,
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                state, parsed = parse_judgment(content, targets)
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as error:
                raise JudgeUnavailable(f"VLM judgment failed: {type(error).__name__}") from error
            responses.append({"image": row["edit"], "status": state, "response": parsed})
            if state == "present":
                return "present", responses
            if state == "unknown":
                overall = "unknown"
        return overall, responses


@torch.inference_mode()
def predict(pipe, scaled_latent, timestep, positive, negative, guidance):
    if guidance > 1:
        result = pipe.unet(
            torch.cat([scaled_latent, scaled_latent]), timestep,
            encoder_hidden_states=torch.cat([negative, positive]),
        ).sample
        uncond, cond = result.chunk(2)
        return uncond + guidance * (cond - uncond)
    return pipe.unet(scaled_latent, timestep, encoder_hidden_states=positive).sample


@torch.inference_mode()
def proxy_order(pipe, controller, pairs, retains, args):
    """One initial leave-one-group-out ranking on matched original trajectories."""
    if getattr(args, "architecture", "sd1") != "sd1":
        from selection_proxy import proxy_order_generic
        return proxy_order_generic(pipe, controller, pairs, retains, args)
    cache = ConditionCache(pipe, "sd1", args)
    device = next(pipe.unet.parameters()).device
    dtype = next(pipe.unet.parameters()).dtype

    def encode(text):
        return cache.get(text).tokens.unsqueeze(0).to(device=device, dtype=dtype)

    negative = encode(args.negative_prompt)
    chosen = {}
    for pair in pairs:
        chosen.setdefault(pair.concept, pair)
    controller.apply(set())
    probes = []
    specs = [(p.target_prompt, p.anchor_prompt, p.concept) for p in chosen.values()]
    specs += [(p, None, None) for p in retains[:args.probe_retains]]
    if not any(kind is None for _, _, kind in specs):
        specs.append(("a photo of a tree.", None, None))
    try:
        for number, (prompt, reference, concept) in enumerate(specs):
            scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
            scheduler.set_timesteps(args.probe_steps, device=device)
            indices = set(torch.linspace(0, args.probe_steps - 1, args.probe_points).round().int().tolist())
            shape = (1, int(pipe.unet.config.in_channels),
                     args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)
            latent = torch.randn(
                shape, generator=torch.Generator(device="cpu").manual_seed(args.seed + number),
            ).to(device=device, dtype=dtype) * scheduler.init_noise_sigma
            positive = encode(prompt)
            for index, timestep in enumerate(scheduler.timesteps):
                scaled = scheduler.scale_model_input(latent, timestep)
                original = predict(pipe, scaled, timestep, positive, negative, args.guidance_scale)
                if index in indices:
                    safe = predict(pipe, scaled, timestep, encode(reference), negative, args.guidance_scale) if reference is not None else original
                    denominator = float((original.float() - safe.float()).square().mean())
                    if concept is None or denominator > 1e-10:
                        probes.append({
                            "latent": scaled.detach().clone(), "t": timestep,
                            "positive": positive, "reference": safe.float().detach().clone(),
                            "denom": max(denominator, 1e-10), "concept": concept,
                        })
                latent = scheduler.step(original, timestep, latent).prev_sample

        def evaluate(active):
            controller.apply(active)
            errors, keep = defaultdict(list), []
            for item in probes:
                output = predict(
                    pipe, item["latent"], item["t"], item["positive"], negative,
                    args.guidance_scale,
                ).float()
                error = float((output - item["reference"]).square().mean())
                if item["concept"] is None:
                    keep.append(error)
                else:
                    errors[item["concept"]].append(error / item["denom"])
            means = {c: sum(v) / len(v) for c, v in errors.items()}
            if set(means) != set(chosen):
                raise ValueError("Proxy prompts have no target-anchor contrast for at least one concept.")
            return means, sum(keep) / max(len(keep), 1)

        full = set(controller.groups)
        baseline, damage = evaluate(full)
        scores = []
        for index, group in enumerate(controller.groups, start=1):
            print(f"[selection proxy] {index}/{len(controller.groups)}: {group}", flush=True)
            residual, trial_damage = evaluate(full - {group})
            scores.append({
                "group": group,
                "risk": max(residual[c] - baseline[c] for c in baseline),
                "preservation_gain": damage - trial_damage,
                "parameter_count": sum(controller.edited[k].numel() for k in controller.groups[group]),
            })
        scores.sort(key=lambda r: (r["risk"], -r["preservation_gain"], -r["parameter_count"], r["group"]))
        return [r["group"] for r in scores], scores
    finally:
        controller.apply(controller.groups)


def run_selection(pipe, controller, pairs, retains, records, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    order, scores = [], []
    judge = ManualJudge() if args.selection_judge == "manual" else VisionJudge(
        args.judge_base_url, args.judge_model, args.judge_key_env,
        args.judge_timeout, args.target_description,
    )
    history = {
        "order": order, "proxy_scores": scores, "rounds": [],
        "stopping_rule": "first_target_reappearance_restore_last_clear",
    }
    active = set(controller.groups)

    def persist():
        (directory / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    for index in range(len(controller.groups) + 1):
        # Check all-updated feasibility BEFORE spending work on proxy ranking.
        if index == 1:
            if args.selection_order == "proxy":
                order, scores = proxy_order(pipe, controller, pairs, retains, args)
            else:
                order = list(controller.groups)
                if args.selection_order == "reverse":
                    order.reverse()
                if args.selection_order == "random":
                    random.Random(args.seed).shuffle(order)
            history["order"], history["proxy_scores"] = order, scores
            persist()
        removed = None if index == 0 else order[index - 1]
        trial = active if removed is None else active - {removed}
        controller.apply(trial)
        round_dir = directory / f"{index:03d}"
        print(f"[selection] round={index}, removed={removed}, groups={len(trial)}", flush=True)
        try:
            images = render_selection(pipe, records, round_dir, args)
        except BaseException:
            controller.apply(active)
            history["status"] = "generation_interrupted"
            history["active_groups"] = sorted(active)
            persist()
            raise
        try:
            status, detail = judge.evaluate(round_dir, images, args.targets)
        except JudgeUnavailable as error:
            status, detail = "unknown", {"error": str(error)}
        history["rounds"].append({
            "round": index, "removed": removed, "active_groups": sorted(trial),
            "status": status, "judgment": detail,
        })
        print(f"[selection] judgment={status}", flush=True)
        persist()
        if status != "clear":
            controller.apply(active)
            history["status"] = (
                "initial_not_clear" if index == 0 else
                "target_reappeared" if status == "present" else "judge_inconclusive"
            )
            print("[selection] Stopping; current trial was not accepted. Restored previous weights.", flush=True)
            break
        active = set(trial)
    else:
        history["status"] = "all_updates_removed_check_judge"
    controller.apply(active)
    history["active_groups"] = sorted(active)
    history["initial_verified_clear"] = history["rounds"][0]["status"] == "clear"
    persist()
    return active, history


DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "su_prompts.json"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def group_counts(count, minimum_fraction=0.5, maximum_fraction=0.75):
    """Keep the requested ratio bounds, including for small entry-only models."""
    if count <= 0 or not 0 < minimum_fraction <= maximum_fraction <= 1:
        raise ValueError("SU fractions must satisfy 0 < minimum <= maximum <= 1, with nonempty groups.")
    minimum, maximum = math.ceil(count * minimum_fraction), math.floor(count * maximum_fraction)
    if minimum > maximum:
        raise ValueError(
            f"No integer group count fits SU fractions {minimum_fraction:g}..{maximum_fraction:g} "
            f"with {count} editable group(s). For a single entry, explicitly use --su-max-fraction 1."
        )
    return minimum, maximum


def middle_plan(groups, minimum_fraction=0.5, maximum_fraction=0.75):
    """Order U-Net attention layers, then expand around its mid_block.

    Diffusers registers up_blocks before mid_block, so named_modules() order is
    not a forward order. Equidistant ties prefer the up side, deterministically.
    Fractions are ceil(50% N) and floor(75% N), counted in attn2 layers, not K/V
    matrices. Non-layer global entries (e.g. SDXL pooled text) stay original.
    """
    phase = {"down_blocks": 0, "mid_block": 1, "up_blocks": 2}

    def key(name):
        section = name.split(".")[0]
        if section not in phase or not name.endswith("attn2"):
            raise ValueError(f"Not a standard U-Net cross-attention layer: {name}")
        return phase[section], tuple(int(x) for x in re.findall(r"\d+", name))

    layers = [name for name in groups
              if name.split(".")[0] in phase and name.endswith("attn2")]
    ordered = sorted(layers, key=key)
    if not ordered:
        raise ValueError(
            "Middle-layer SU requires editable attention layers ordered by depth. "
            "This adapter exposes only conditioning entries; use --selection-method vlm or legacy."
        )
    mid = [i for i, name in enumerate(ordered) if name.startswith("mid_block.")]
    if not mid:
        raise ValueError("Middle selection requires a U-Net mid_block with cross-attention.")
    center = (mid[0] + mid[-1]) / 2
    indices = sorted(range(len(ordered)), key=lambda i: (abs(i - center), -i))
    expansion = [ordered[i] for i in indices]
    minimum, maximum = group_counts(len(ordered), minimum_fraction, maximum_fraction)
    candidates = []
    for count in range(minimum, maximum + 1):
        active = set(expansion[:count])
        candidates.append({
            "id": f"layers_{count:02d}_of_{len(ordered):02d}",
            "group_count": count, "layer_count": count, "fraction": count / len(ordered),
            "active_groups": [name for name in ordered if name in active],
            "added_group": expansion[count - 1] if count > minimum else None,
        })
    middle = set(expansion[:minimum])
    return {
        "unit_kind": "attention_layer", "basis": "unet_mid_block",
        "total_groups": len(ordered), "total_layers": len(ordered), "ordered_groups": ordered,
        "excluded_groups": [name for name in groups if name not in layers],
        "minimum_fraction": minimum_fraction, "maximum_fraction": maximum_fraction,
        "center_groups": [ordered[i] for i in mid], "expansion_order": expansion,
        "minimum_layers": minimum, "maximum_layers": maximum,
        "middle_groups": [name for name in ordered if name in middle],
        "candidates": candidates,
    }


def entry_plan(groups, minimum_fraction=0.5, maximum_fraction=0.75):
    """Bounded candidates for parallel text-entry groups, without claiming depth.

    Each group seeds one chain; grow cyclically in adapter declaration order.
    Deduplicate equal subsets. This is at most O(n^2) candidates, not 2**n.
    For SD3's two entries, 50%..75% gives both singleton candidates.
    """
    ordered = list(groups)
    minimum, maximum = group_counts(len(ordered), minimum_fraction, maximum_fraction)
    candidates, seen = [], set()
    for count in range(minimum, maximum + 1):
        for start in range(len(ordered)):
            active = frozenset(ordered[(start + i) % len(ordered)] for i in range(count))
            if active in seen:
                continue
            seen.add(active)
            candidates.append({
                "id": f"groups_{count:02d}_of_{len(ordered):02d}_{start:02d}",
                "group_count": count, "fraction": count / len(ordered),
                "active_groups": [name for name in ordered if name in active],
            })
    return {
        "unit_kind": "conditioning_entry", "basis": "parallel_entry_chains",
        "total_groups": len(ordered), "ordered_groups": ordered, "excluded_groups": [],
        "minimum_fraction": minimum_fraction, "maximum_fraction": maximum_fraction,
        "minimum_groups": minimum, "maximum_groups": maximum, "candidates": candidates,
    }


def selection_plan(groups, method, minimum_fraction=0.5, maximum_fraction=0.75):
    """Model-independent dispatch based on actual editable group topology."""
    has_layers = any(name.endswith("attn2") and name.split(".")[0] in {
        "down_blocks", "mid_block", "up_blocks",
    } for name in groups)
    if method == "middle" or has_layers:
        return middle_plan(groups, minimum_fraction, maximum_fraction)
    return entry_plan(groups, minimum_fraction, maximum_fraction)


def calibration_records(targets, task, template_count, samples, seed, path=None):
    """New, small prompt set, separate from CA fitting and paper evaluation."""
    data = json.loads(Path(path or DEFAULT_PROMPTS).read_text(encoding="utf-8"))
    templates = data.get(task) if isinstance(data, dict) else None
    if not isinstance(templates, list) or not templates or not all(
        isinstance(t, str) and "{concept}" in t for t in templates
    ):
        raise ValueError("SU prompts must be a nonempty list containing {concept} in every template.")
    if not 1 <= template_count <= len(templates) or samples <= 0:
        raise ValueError(f"SU template count must be 1..{len(templates)} and samples positive.")
    records = []
    for concept in targets:
        for index, template in enumerate(templates[:template_count]):
            for sample in range(samples):
                image_seed = seed + 100_000 + index * samples + sample
                try:
                    prompt = template.format(concept=concept)
                except (KeyError, ValueError, IndexError) as error:
                    raise ValueError("SU prompts only support the {concept} placeholder.") from error
                records.append({
                    "concept": concept, "prompt": prompt,
                    "template_index": index, "sample_index": sample, "seed": image_seed,
                    "filename": f"su{index:03d}_s{sample:03d}_seed{image_seed}.png",
                })
    return records


def release_cuda():
    """Call after deleting all live model/controller references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_candidates(pipe, controller, plan, records, directory, args, metadata):
    """Generate every candidate first. No judge is constructed in this phase."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "layer_plan.json", plan)
    write_json(directory / "prompts.json", records)
    candidates = []
    if not plan["candidates"]:
        raise ValueError("No candidate fits the requested selective-update group bounds.")
    for spec in plan["candidates"]:
        controller.apply(spec["active_groups"])
        gallery = directory / spec["id"]
        print(f"[SU:generate] {spec['id']}, images={len(records)}", flush=True)
        rows = render_selection(pipe, records, gallery, args)
        if not rows or len(rows) != len(records):
            raise ValueError("Incomplete SU candidate gallery; refusing empty calibration.")
        candidates.append({**spec, "directory": spec["id"], "records": rows})
    # Parallel-entry candidates need not be nested: save the UNION, not merely
    # the last candidate's weights. This union is storage, never a judged trial.
    controller.apply({name for spec in plan["candidates"] for name in spec["active_groups"]})
    # One superset checkpoint suffices; do not duplicate weights per candidate.
    controller.save(directory / "candidate_weights", {
        **metadata, "selection": {"method": "vlm", "unit_kind": plan["unit_kind"], "status": "awaiting_vlm"},
    })
    state = {"plan": plan, "candidates": candidates, "targets": args.targets}
    write_json(directory / "candidates.json", state)
    return state


def visual_instruction(targets, task, description=""):
    schema = {"concepts": {t: {"present": None, "evidence": "brief visual evidence"} for t in targets}}
    kind = (
        "The concepts are artist styles. Assess resemblance to the artist's visual style, "
        "not whether the artist's face or written name appears."
        if task == "style" else
        "The concepts are specific named characters or objects. A generic dog, mouse or "
        "sponge is not sufficient evidence of the named cartoon character."
    )
    return (
        "Inspect only this image. Decide whether EACH requested concept is visually present. "
        + kind + " Use true when present, false when absent, null when uncertain. "
        "Return ONLY a JSON object with exactly this schema and these concept keys: "
        + json.dumps(schema) + "\nAdditional visual definitions: " + description
    )


class LocalVisionJudge:
    """A local Transformers VLM, loaded only after diffusion has been released."""

    def __init__(self, args):
        # Lazy imports avoid loading/initializing a VLM during candidate generation.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = self.processor = None
        self.args = args
        name = args.su_vlm_model
        dtype = torch.float32 if not args.device.startswith("cuda") else torch.float16
        try:
            self.processor = AutoProcessor.from_pretrained(name, local_files_only=args.local_files_only)
            self.model = AutoModelForImageTextToText.from_pretrained(
                name, torch_dtype=dtype, local_files_only=args.local_files_only,
            ).to(args.device).eval()
        except Exception:
            self.close()
            raise

    @torch.inference_mode()
    def evaluate(self, directory, records, targets):
        if not records:
            raise JudgeUnavailable("No images to judge.")
        responses, overall = [], "clear"
        instruction = visual_instruction(targets, self.args.task, self.args.target_description)
        for row in records:
            content = ""
            with Image.open(directory / row["edit"]) as source:
                picture = source.convert("RGB")
            # Restrict VLM input to limit image tokens/memory independently of SD resolution.
            picture.thumbnail((self.args.su_vlm_image_size, self.args.su_vlm_image_size))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": picture}, {"type": "text", "text": instruction},
            ]}]
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[picture], padding=True, return_tensors="pt")
                inputs = inputs.to(self.args.device)
                generated = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                content = self.processor.batch_decode(
                    generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                state, parsed = parse_judgment(content, targets)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                # Invalid/uncertain output must never count as successful erasure.
                state, parsed = "unknown", {"error": str(error), "raw": content}
            responses.append({"image": row["edit"], "status": state, "response": parsed})
            if state == "present":
                return "present", responses
            if state != "clear":
                overall = "unknown"
        return overall, responses

    def close(self):
        self.model = self.processor = None
        release_cuda()


def judge_candidates(state, directory, args, judge_factory=None):
    """Fewest-group clear candidate; uncertainty never counts as clearance."""
    report = {
        "method": "vlm", "status": "no_feasible_candidate",
        "total_groups": state["plan"]["total_groups"], "unit_kind": state["plan"]["unit_kind"],
        "judge_model": args.su_vlm_model,
        "history": [], "active_groups": [],
    }
    if state["plan"]["unit_kind"] == "attention_layer":
        report["total_layers"] = state["plan"]["total_groups"]
    selected = None
    judge = None
    try:
        judge = (judge_factory or LocalVisionJudge)(args)
        for candidate in sorted(state["candidates"], key=lambda c: c["group_count"]):
            rows = candidate["records"]
            if not rows:
                raise JudgeUnavailable("Empty candidate gallery cannot pass SU.")
            status, evidence = judge.evaluate(directory / candidate["directory"], rows, state["targets"])
            report["history"].append({
                "candidate": candidate["id"], "group_count": candidate["group_count"],
                "status": status, "evidence": evidence,
            })
            write_json(directory / "judgments.json", report)
            print(f"[SU:judge] {candidate['id']}: {status}", flush=True)
            if status == "clear":
                selected = candidate
                report.update(status="selected", selected_candidate=candidate["id"],
                              active_groups=candidate["active_groups"],
                              selected_group_count=candidate["group_count"])
                if report["unit_kind"] == "attention_layer":
                    report["selected_layer_count"] = candidate["group_count"]
                break
    except Exception as error:
        # Preserve all generated candidates; never publish an unverified checkpoint.
        report.update(status="judge_unavailable", error=f"{type(error).__name__}: {error}")
    finally:
        if judge is not None:
            judge.close()
        write_json(directory / "judgments.json", report)
    return selected, report


def publish_candidate(directory, selected, checkpoint, report):
    """Extract selected absolute weights on CPU without reloading SD or VLM."""
    source = directory / "candidate_weights"
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    all_weights = load_file(str(source / "weights.safetensors"), device="cpu")
    keys = {key for name in selected["active_groups"] for key in metadata["groups"][name]}
    weights = {key: all_weights[key] for key in sorted(keys)}
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(checkpoint / "weights.safetensors"))
    metadata.update(active_groups=selected["active_groups"], selection=report)
    write_json(checkpoint / "metadata.json", metadata)
