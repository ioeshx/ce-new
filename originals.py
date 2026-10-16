"""Original images in readable task/concept folders, without hashes or locks."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def concept_folder(concept):
    name = re.sub(r"[^\w-]+", "_", concept.strip().casefold()).strip("_")
    if not name:
        raise ValueError(f"Cannot use concept as a directory name: {concept!r}")
    return name


def original_config(pipe, args):
    from models import backbone

    return {
        "base_model": args.sd_ckpt, "architecture": args.architecture,
        "dtype": str(next(backbone(pipe, args.architecture).parameters()).dtype),
        "device": str(args.device).split(":", 1)[0],
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale, "negative_prompt": args.negative_prompt,
        "height": args.height, "width": args.width, "batch_size": args.batch_size,
        "flux_variant": args.flux_variant, "max_sequence_length": args.max_sequence_length,
        "scheduler": {
            "class": type(pipe.scheduler).__name__,
            "config": {k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")},
        },
    }


def usable_png(path, width, height):
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "PNG" and image.size == (width, height)
    except (OSError, ValueError):
        return False


class OriginalImages:
    def __init__(self, root, task, config):
        if task not in {"instance", "style"}:
            raise ValueError("Original task must be instance or style.")
        self.root = Path(root).expanduser().resolve()
        self.task = task
        self.config = json.loads(json.dumps(config, default=str))

    def render(self, pipe, records, args):
        from sampling import generate_images

        groups = defaultdict(list)
        names = {}
        paths = [None] * len(records)
        for index, row in enumerate(records):
            name = concept_folder(row["concept"])
            if name in names and names[name] != row["concept"]:
                raise ValueError(f"Concepts map to the same original directory: {names[name]!r}, {row['concept']!r}")
            names[name] = row["concept"]
            groups[name].append((index, row))

        for name, items in groups.items():
            directory = self.root / self.task / name
            directory.mkdir(parents=True, exist_ok=True)
            manifest = [
                {**{k: row[k] for k in ("concept", "prompt", "seed", "template_index", "sample_index", "prompt_id") if k in row},
                 "filename": f"{index:06d}.png"}
                for index, (_, row) in enumerate(items)
            ]
            config = {"task": self.task, "concept": items[0][1]["concept"],
                      "image_count": len(items), "sampling": self.config}
            config_path, manifest_path = directory / "config.json", directory / "manifest.json"
            unchanged = read_json(config_path) == config and read_json(manifest_path) == manifest
            if not unchanged:
                # Only this task/concept's generated PNGs are replaced.
                for path in directory.glob("*.png"):
                    path.unlink()
                print(f"[original:{self.task}/{name}] config/prompts changed; regenerating originals.", flush=True)
            write_json(config_path, config)
            write_json(manifest_path, manifest)
            pending = [(row, directory / entry["filename"])
                       for (_, row), entry in zip(items, manifest)
                       if not usable_png(directory / entry["filename"], args.width, args.height)]
            print(f"[original:{self.task}/{name}] total={len(items)}, pending={len(pending)}, "
                  f"reused={len(items) - len(pending)}; directory={directory}", flush=True)
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                images = generate_images(pipe, [row for row, _ in batch], args)
                if len(images) != len(batch):
                    raise RuntimeError("Pipeline returned fewer original images than requested.")
                for (_, path), image in zip(batch, images):
                    temporary = path.with_suffix(".tmp.png")
                    image.save(temporary, format="PNG")
                    temporary.replace(path)
            for (index, _), entry in zip(items, manifest):
                paths[index] = directory / entry["filename"]
        return paths


def validate_original_reference(path, row, *, sampling=None, task=None, known=None):
    """Check direct references against the public folder's plain JSON files."""
    known = {} if known is None else known
    directory = path.parent
    if directory not in known:
        config = read_json(directory / "config.json")
        manifest = read_json(directory / "manifest.json")
        if not isinstance(config, dict) or not isinstance(manifest, list):
            raise ValueError(f"Shared original needs config.json and manifest.json: {directory}")
        known[directory] = config, {entry["filename"]: entry for entry in manifest}
    config, entries = known[directory]
    entry = entries.get(path.name)
    if (entry is None or config.get("task") != task
            or any(entry.get(key) != row.get(key) for key in ("concept", "prompt", "seed"))):
        raise ValueError(f"Shared original does not match this concept/prompt/seed: {path}")
    settings = config.get("sampling", {})
    for name, expected in (sampling or {}).items():
        actual_name = "base_model" if name == "sd_ckpt" else name
        if actual_name not in settings or name == "dtype" and expected == "auto":
            continue
        actual = settings[actual_name]
        if name == "dtype":
            actual, expected = str(actual).removeprefix("torch."), str(expected).removeprefix("torch.")
        elif name == "device":
            actual, expected = str(actual).split(":", 1)[0], str(expected).split(":", 1)[0]
        if actual != expected:
            raise ValueError(f"Shared original config differs from this experiment ({name}): {directory}")
