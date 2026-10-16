"""Per-concept CLIP and FID for paired original/edited image manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CLIP = "openai/clip-vit-base-patch16"


def load_manifest(directory, *, checkpoint=None, targets=None, task=None):
    from evaluation import concept_key

    directory = Path(directory).expanduser().resolve()
    if not (directory / "manifest.json").is_file() and (directory / "images/manifest.json").is_file():
        directory = directory / "images"
    records = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Image manifest must be a nonempty list.")
    plan_path = directory / "dataset.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    if plan:
        source = [{k: v for k, v in row.items() if k not in {"original", "edit", "compare"}}
                  for row in records if isinstance(row, dict)]
        digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        if len(records) != plan["record_count"] or digest != plan["records_sha256"]:
            raise ValueError("Image manifest does not match the full planned dataset; finish/resume sampling first.")
    if checkpoint is None:
        checkpoint = plan.get("checkpoint")
    if checkpoint is None and (directory.parent / "checkpoint/metadata.json").is_file():
        checkpoint = directory.parent / "checkpoint"
    metadata = {}
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        checkpoint = checkpoint.parent if checkpoint.is_file() else checkpoint
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if plan:
            from evaluation import file_sha256
            if (file_sha256(checkpoint / "metadata.json") != plan["checkpoint_metadata_sha256"]
                    or file_sha256(checkpoint / "weights.safetensors") != plan["checkpoint_weights_sha256"]):
                raise ValueError("Evaluation checkpoint differs from the one used to generate these images.")
    targets = targets if targets is not None else metadata.get("targets", plan.get("targets"))
    task = task or metadata.get("task", plan.get("task"))
    if not targets or task not in {"instance", "style"}:
        raise ValueError("Supply --checkpoint or both --target-concepts and --task to identify target/non-target groups.")
    target_keys = {concept_key(t) for t in targets}
    groups, seen = defaultdict(list), {"original": set(), "edit": set()}
    shared_originals = {}
    for index, row in enumerate(records):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("concept", "prompt", "original", "edit")):
            raise ValueError(f"Manifest row {index} needs concept, prompt, original and edit.")
        for mode in seen:
            path = (directory / row[mode]).resolve()
            if not path.is_file() or mode == "edit" and not path.is_relative_to(directory):
                raise ValueError(f"Missing image or path outside the dataset: {row[mode]}")
            if mode == "original" and not path.is_relative_to(directory):
                from originals import validate_original_reference
                validate_original_reference(path, row, sampling=plan.get("sampling"), task=task,
                                            known=shared_originals)
            if path in seen[mode]:
                raise ValueError(f"Duplicate {mode} image in manifest: {row[mode]}")
            seen[mode].add(path)
        row = dict(row)
        row["is_target"] = concept_key(row["concept"]) in target_keys
        groups[concept_key(row["concept"])].append(row)
    missing = target_keys - groups.keys()
    if missing:
        raise ValueError(f"No evaluation images for target concepts: {sorted(missing)}")
    return directory, groups, {"task": task, "targets": targets, "checkpoint": str(checkpoint) if checkpoint else None}


class CLIPScorer:
    def __init__(self, model_name, device, local_files_only=False):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only, use_fast=False)
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_files_only, torch_dtype=torch.float32,
        ).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def score(self, paths, prompts):
        import torch
        from PIL import Image

        length = self.model.config.text_config.max_position_embeddings
        tokenized = self.processor.tokenizer(prompts, truncation=False)["input_ids"]
        if any(len(x) > length for x in tokenized):
            raise ValueError(f"A CLIP prompt exceeds {length} tokens; shorten it instead of silently truncating.")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            image = torch.nn.functional.normalize(output.image_embeds.float(), dim=-1)
            text = torch.nn.functional.normalize(output.text_embeds.float(), dim=-1)
            # Match util/clip_score_cal.py: 100*cosine, without zero clipping.
            scores = 100 * (image * text).sum(-1)
        return scores.cpu().numpy()


class FIDExtractor:
    def __init__(self, device, local_files_only=False):
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3, FeatureExtractorInceptionV3

        weight = None
        if local_files_only:
            weight = Path(torch.hub.get_dir()) / "checkpoints" / URL_INCEPTION_V3.rsplit("/", 1)[-1]
            if not weight.is_file():
                raise ValueError(f"Offline FID needs the standard torch-fidelity weights at {weight}.")
        self.device = device
        self.model = FeatureExtractorInceptionV3(
            "inception-v3-compat", ["2048"],
            feature_extractor_weights_path=str(weight) if weight else None,
            verbose=False,
        ).to(device).eval()

    def extract(self, paths, batch_size, workers):
        import torch
        from torch_fidelity.datasets import ImagesPathDataset
        from torch.utils.data import DataLoader
        import numpy as np

        loader = DataLoader(ImagesPathDataset(paths), batch_size=batch_size, shuffle=False,
                            num_workers=workers, drop_last=False)
        values = []
        with torch.inference_mode():
            for index, images in enumerate(loader):
                # Same uint8 input and TensorFlow-compatible resize as the old
                # torch_fidelity.calculate_metrics call; do not pre-resize.
                values.append(self.model(images.to(self.device))[0].float().cpu().numpy())
                print(f"[FID features] {min((index + 1) * batch_size, len(paths))}/{len(paths)}", flush=True)
        return np.concatenate(values)


def fid_from_features(original, edited, method="auto"):
    """Standard sample-Gaussian FID, optionally using an equivalent thin factor.

    For centered row features A and B, the covariance square-root trace equals
    ||A @ B.T||_* / sqrt((N-1)(M-1)). This avoids a 2048-D matrix square root
    when there are only 300/800 images per concept. Features remain 2048-D.
    """
    import numpy as np

    a, b = np.asarray(original, dtype=np.float64), np.asarray(edited, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(len(a), len(b)) < 2:
        raise ValueError("FID needs at least two images per distribution and matching feature dimensions.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite Inception features.")
    if method == "auto":
        method = "lowrank" if max(len(a), len(b)) < a.shape[1] else "covariance"
    am, bm = a.mean(0), b.mean(0)
    if method == "lowrank":
        ac, bc = a - am, b - bm
        trace_a = (ac * ac).sum() / (len(a) - 1)
        trace_b = (bc * bc).sum() / (len(b) - 1)
        cross = (ac @ bc.T) / ((len(a) - 1) * (len(b) - 1)) ** 0.5
        root_trace = np.linalg.svd(cross, compute_uv=False).sum()
        value = float(((am - bm) ** 2).sum() + trace_a + trace_b - 2 * root_trace)
    elif method == "covariance":
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        value = float(fid_statistics_to_metric(
            {"mu": am, "sigma": np.atleast_2d(np.cov(a, rowvar=False))},
            {"mu": bm, "sigma": np.atleast_2d(np.cov(b, rowvar=False))}, False,
        )["frechet_inception_distance"])
    else:
        raise ValueError("FID method must be auto, lowrank or covariance.")
    if not np.isfinite(value) or value < -1e-5:
        raise ValueError(f"Invalid numerical FID: {value}")
    return max(0.0, value), method


def feature_fingerprint(paths):
    entries = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    import importlib.metadata
    return hashlib.sha256(json.dumps(["torch-fidelity-inception-v3-compat-2048",
                                     importlib.metadata.version("torch-fidelity"), entries]).encode()).hexdigest()


def cached_features(paths, path, extract, *, use_cache=True, refresh=False):
    import numpy as np

    fingerprint = feature_fingerprint(paths)
    if use_cache and not refresh and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as saved:
                values = saved["features"]
                if (str(saved["fingerprint"].item()) == fingerprint
                        and values.shape == (len(paths), 2048) and np.isfinite(values).all()):
                    print(f"[FID cache] {path.name}", flush=True)
                    return values
        except (OSError, ValueError, KeyError):
            pass
    values = np.asarray(extract(paths), dtype=np.float32)
    if values.shape != (len(paths), 2048) or not np.isfinite(values).all():
        raise ValueError("Expected finite 2048-D standard Inception features for every image.")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, features=values, fingerprint=fingerprint)
        temporary.replace(path)
    return values


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def release_model(backend):
    if backend is not None:
        del backend.model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_metrics(image_dir, *, output_dir=None, checkpoint=None, targets=None, task=None,
                    device="cuda", batch_size=32, workers=0, clip_model=DEFAULT_CLIP,
                    local_files_only=False, fid_method="auto", use_cache=True, refresh_cache=False,
                    requested=("clip", "fid")):
    import numpy as np
    import torch
    from sampling import slug

    if batch_size <= 0 or workers < 0 or not requested or set(requested) - {"clip", "fid"}:
        raise ValueError("Invalid metric names, batch size or worker count.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu for metrics.")
    directory, groups, context = load_manifest(image_dir, checkpoint=checkpoint, targets=targets, task=task)
    if "fid" in requested and any(len(rows) < 2 for rows in groups.values()):
        raise ValueError("FID needs at least two original and edited images for EACH concept.")
    suffix = hashlib.sha1(str(directory).encode()).hexdigest()[:8]
    output = Path(output_dir).expanduser().resolve() if output_dir else ROOT / "metric" / f"{directory.name}-{suffix}"
    if output == directory:
        raise ValueError("Metric output must differ from the image dataset directory.")
    output.mkdir(parents=True, exist_ok=True)
    per_image = [dict(row) for rows in groups.values() for row in rows]
    clip_revision = None
    if "clip" in requested:
        backend = CLIPScorer(clip_model, device, local_files_only)
        clip_revision = backend.revision
        try:
            for mode in ("original", "edit"):
                for start in range(0, len(per_image), batch_size):
                    chunk = per_image[start:start + batch_size]
                    scores = backend.score([directory / r[mode] for r in chunk], [r["prompt"] for r in chunk])
                    if len(scores) != len(chunk) or not np.isfinite(scores).all():
                        raise ValueError("CLIP returned invalid/missing scores.")
                    for row, score in zip(chunk, scores):
                        row[f"clip_{mode}"] = float(score)
                    print(f"[CLIP:{mode}] {min(start + len(chunk), len(per_image))}/{len(per_image)}", flush=True)
        finally:
            release_model(backend)
        for row in per_image:
            row["clip_delta"] = row["clip_edit"] - row["clip_original"]

    results, extractor = [], None
    try:
        def extract(paths):
            nonlocal extractor
            if extractor is None:
                extractor = FIDExtractor(device, local_files_only)
            return extractor.extract(paths, batch_size, workers)

        offset = 0
        for key, rows in groups.items():
            scored_rows = per_image[offset:offset + len(rows)]
            offset += len(rows)
            result = {"concept": rows[0]["concept"], "role": "target" if rows[0]["is_target"] else "non_target", "n": len(rows)}
            if "clip" in requested:
                for name in ("clip_original", "clip_edit", "clip_delta"):
                    result[name] = float(np.mean([r[name] for r in scored_rows]))
            if "fid" in requested:
                features = [cached_features(
                    [directory / r[mode] for r in rows], output / "cache" / f"{slug(key)}-{mode}.npz", extract,
                    use_cache=use_cache, refresh=refresh_cache,
                ) for mode in ("original", "edit")]
                result["fid"], result["fid_method"] = fid_from_features(*features, method=fid_method)
            results.append(result)
            print(f"[metric] {result}", flush=True)
    finally:
        release_model(extractor)
    aggregates = {}
    for role in ("target", "non_target", "all"):
        selected = [row for row in results if role == "all" or row["role"] == role]
        if not selected:
            aggregates[role] = None
            continue
        entry = {"concept_count": len(selected), "image_count_per_mode": sum(r["n"] for r in selected)}
        if "fid" in requested:
            entry["fid_macro"] = float(np.mean([r["fid"] for r in selected]))
        if "clip" in requested:
            for name in ("clip_original", "clip_edit", "clip_delta"):
                entry[f"{name}_macro"] = float(np.mean([r[name] for r in selected]))
                entry[f"{name}_micro"] = float(np.average([r[name] for r in selected], weights=[r["n"] for r in selected]))
        aggregates[role] = entry
    import importlib.metadata
    report = {
        "schema": "casu2-metrics-v1", **context, "image_dir": str(directory),
        "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        "definitions": {
            "clip": "mean(100*cos(image_embedding, full_generation_prompt_embedding)); no zero clipping",
            "fid": "2048-D torch-fidelity inception-v3-compat: original vs edited, separately per concept",
            "fid_macro": "arithmetic mean of per-concept FIDs; NOT pooled-distribution FID",
        },
        "config": {"requested": list(requested), "clip_model": clip_model, "clip_revision": clip_revision,
                   "fid_dims": 2048, "fid_method": fid_method, "device": str(device),
                   "batch_size": batch_size, "workers": workers},
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "transformers", "torch-fidelity", "numpy", "Pillow",
        )},
        "per_concept": results, "aggregate": aggregates,
    }
    temporary = output / "results.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "results.json")
    write_csv(output / "per_concept.csv", results)
    if "clip" in requested:
        write_csv(output / "per_image_clip.csv", per_image)
    else:
        (output / "per_image_clip.csv").unlink(missing_ok=True)
    print(f"[metric output] {output}", flush=True)
    return report


def metrics_main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py metrics", description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: casu-2/metric/<dataset-name>-<path-hash>.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-concepts", help="Optional override; comma-separated.")
    parser.add_argument("--task", choices=["instance", "style"])
    parser.add_argument("--metrics", default="clip,fid")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP, help="HF CLIP model ID or local directory.")
    parser.add_argument("--fid-method", choices=["auto", "lowrank", "covariance"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)
    try:
        import torch
        if args.cpu_threads <= 0:
            raise ValueError("CPU threads must be positive.")
        torch.set_num_threads(args.cpu_threads)
        from prompts import split_concepts
        compute_metrics(
            args.image_dir, output_dir=args.output_dir, checkpoint=args.checkpoint,
            targets=split_concepts(args.target_concepts) if args.target_concepts else None, task=args.task,
            device=args.device, batch_size=args.batch_size, workers=args.workers, clip_model=args.clip_model,
            local_files_only=args.local_files_only, fid_method=args.fid_method,
            use_cache=args.feature_cache, refresh_cache=args.refresh_cache,
            requested=tuple(x.strip() for x in args.metrics.split(",")),
        )
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    return 0
