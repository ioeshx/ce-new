"""Matched-seed before/after sampling without a second backbone copy."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

import torch
from PIL import Image, ImageDraw

from prompts import choose_templates
from models import backbone, pipeline_device


def slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")[:70] or "concept"
    return f"{clean}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def sample_records(concepts, task, template_count, samples, seed, offset=0):
    records = []
    for concept in concepts:
        for index, template in choose_templates(task, template_count, offset):
            for sample in range(samples):
                records.append({
                    "concept": concept, "prompt": template.format(concept),
                    "template_index": index, "sample_index": sample,
                    "seed": seed + sample,
                    "filename": f"t{index:03d}_s{sample:03d}_seed{seed + sample}.png",
                })
    return records


def pipeline_arguments(pipe, records, args):
    """Native 4-D noise layouts for SD models and FLUX.2-klein."""
    architecture = getattr(args, "architecture", "sd1")
    model = backbone(pipe, architecture)
    device = pipeline_device(pipe)
    dtype = next(model.parameters()).dtype
    channels = int(model.config.in_channels)
    scale = pipe.vae_scale_factor * (2 if architecture == "flux" else 1)
    shape = (
        channels,
        args.height // scale, args.width // scale,
    )
    noise = torch.stack([
        torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(r["seed"]))
        for r in records
    ]).to(device=device, dtype=dtype)
    kwargs = dict(
        prompt=[r["prompt"] for r in records], height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale, latents=noise,
        generator=[torch.Generator(device="cpu").manual_seed(r["seed"]) for r in records],
    )
    if architecture in {"sd3", "flux"}:
        kwargs["max_sequence_length"] = args.max_sequence_length
    # Klein packs its patched 4-D noise internally and uses no negative prompt.
    if architecture != "flux":
        kwargs["negative_prompt"] = [args.negative_prompt] * len(records)
    return kwargs


@torch.inference_mode()
def generate_images(pipe, records, args):
    """Explicit initial noise makes before/after independent of batching/state."""
    if not records:
        return []
    result = pipe(**pipeline_arguments(pipe, records, args))
    return list(result.images)


def write_gallery(directory, records, columns):
    rows = []
    for row in records:
        pictures = "".join(
            f'<td>{html.escape(name)}<br><a href="{quote(row[name])}">'
            f'<img loading="lazy" width="320" src="{quote(row[name])}"></a></td>'
            for name in columns if name in row
        )
        rows.append(
            f'<tr><td>{html.escape(row["prompt"])}<br>seed={row["seed"]}</td>{pictures}</tr>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>CASU-2 images</title>'
        '<style>body{font-family:sans-serif}td{padding:8px;vertical-align:top}'
        'table{border-collapse:collapse}tr{border-bottom:1px solid #ccc}</style>'
        '<p>Original and edited images use identical initial noise and settings.</p>'
        '<table>' + "\n".join(rows) + "</table>"
    )
    (directory / "index.html").write_text(page, encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def link_originals(directory, paths, original_directory):
    """Expose shared originals through one relative directory symlink."""
    original_directory = Path(original_directory).resolve()
    relative = [(Path("original") / path.relative_to(original_directory)).as_posix() for path in paths]
    link = directory / "original"
    if link.is_symlink() and link.resolve() != original_directory:
        link.unlink()
    if not link.is_symlink():
        link.symlink_to(os.path.relpath(original_directory, directory), target_is_directory=True)
    return relative


def render_selection(pipe, records, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        images = generate_images(pipe, chunk, args)
        for row, image in zip(chunk, images):
            relative = Path(slug(row["concept"])) / row["filename"]
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            image.save(directory / relative)
            result.append({**row, "edit": relative.as_posix()})
    write_gallery(directory, result, ["edit"])
    return result


def remove_legacy_originals(directory, records):
    """Remove only the old per-row original paths, after publishing shared references."""
    removed, folders = 0, set()
    for row in records:
        legacy = directory / slug(row["concept"]) / "original" / row["filename"]
        if legacy.is_file() and legacy.resolve().is_relative_to(directory.resolve()):
            legacy.unlink()
            removed += 1
            folders.add(legacy.parent)
    for folder in folders:
        try:
            folder.rmdir()
        except OSError:
            pass  # Preserve any other files not belonging to the current dataset.
    if removed:
        print(f"[original paths] Removed {removed} legacy experiment-local originals.", flush=True)
    return removed


def render_comparison(pipe, controller, active, records, directory, args, modes=("original", "edit"),
                      *, resume=False, comparisons=True, originals=None):
    directory.mkdir(parents=True, exist_ok=True)
    results = [dict(row) for row in records]
    try:
        for mode in modes:
            controller.apply(set() if mode == "original" else active)
            if mode == "original" and originals is not None:
                paths = originals.render(pipe, records, args)
                paths = link_originals(directory, paths, originals.root / originals.task)
                for index, path in enumerate(paths):
                    results[index][mode] = path
                continue
            pending = []
            for index, row in enumerate(records):
                relative = Path(slug(row["concept"])) / mode / row["filename"]
                path = directory / relative
                results[index][mode] = relative.as_posix()
                usable = False
                if resume and path.is_file():
                    try:
                        with Image.open(path) as existing:
                            existing.load()
                            usable = existing.format == "PNG" and existing.size == (args.width, args.height)
                    except (OSError, ValueError):
                        pass
                if not usable:
                    pending.append((row, path))
            print(f"[sample:{mode}] total={len(records)}, pending={len(pending)}, reused={len(records) - len(pending)}", flush=True)
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                chunk = [row for row, _ in batch]
                images = generate_images(pipe, chunk, args)
                if len(images) != len(batch):
                    raise RuntimeError("Pipeline returned fewer images than requested.")
                for (_, path), image in zip(batch, images):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp.png")
                    image.save(temporary)
                    temporary.replace(path)
                print(f"[sample:{mode}] {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
        if comparisons and "original" in modes and "edit" in modes:
            for row in results:
                with Image.open(directory / row["original"]) as left, Image.open(directory / row["edit"]) as right:
                    combined = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 24), "white")
                    combined.paste(left, (0, 24))
                    combined.paste(right, (left.width, 24))
                    draw = ImageDraw.Draw(combined)
                    draw.text((8, 5), "original", fill="black")
                    draw.text((left.width + 8, 5), "edited", fill="black")
                    relative = Path(slug(row["concept"])) / "compare" / row["filename"]
                    (directory / relative).parent.mkdir(parents=True, exist_ok=True)
                    combined.save(directory / relative)
                    row["compare"] = relative.as_posix()
        write_gallery(directory, results, list(modes))
        if originals is not None and "original" in modes:
            remove_legacy_originals(directory, records)
    finally:
        controller.apply(active)
    return results
