"""First-failure backward deletion with manual or vision-language judging."""

from __future__ import annotations

import base64
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import urllib.error
import urllib.request

import torch

from sampling import render_selection
from sd import TextCache


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
    cache = TextCache(pipe)
    device = next(pipe.unet.parameters()).device
    dtype = next(pipe.unet.parameters()).dtype

    def encode(text):
        return cache.get(text).hidden.unsqueeze(0).to(device=device, dtype=dtype)

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
