"""Paper evaluation sets and provenance for resumable paired sampling."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

from sampling import sample_records

PAPER_CONTENTS = {
    "instance": ["Hello Kitty", "Mickey", "Pikachu", "Snoopy", "Spongebob"],
    "style": ["Van Gogh", "Monet", "Picasso"],
}


def concept_key(name):
    key = " ".join(name.casefold().split())
    return {
        "mickey mouse": "mickey", "sponge bob": "spongebob",
        "claude monet": "monet", "pablo picasso": "picasso",
        "vincent van gogh": "van gogh",
    }.get(key, key)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_records(args, metadata):
    from prompts import split_concepts

    task, targets = metadata["task"], metadata["targets"]
    if args.prompt_csv:
        if args.sample_template_count or args.sample_template_offset:
            raise ValueError("CSV sampling uses all its rows; template count/offset must be 0.")
        requested = {concept_key(x) for x in split_concepts(args.contents)} if args.contents else None
        records, found = [], set()
        with args.prompt_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"concept", "prompt"} <= set(reader.fieldnames or []):
                raise ValueError("Prompt CSV needs concept and prompt columns.")
            for index, row in enumerate(reader):
                concept, prompt = row["concept"].strip(), row["prompt"].strip()
                if not concept or not prompt:
                    raise ValueError(f"Empty concept/prompt in CSV row {index + 2}.")
                key = concept_key(concept)
                if requested is not None and key not in requested:
                    continue
                found.add(key)
                for sample in range(args.num_samples):
                    records.append({
                        "concept": concept, "prompt": prompt, "template_index": index,
                        "prompt_id": row.get("id", str(index)), "sample_index": sample,
                        "seed": args.seed + sample,
                        "filename": f"p{index:06d}_s{sample:03d}_seed{args.seed + sample}.png",
                    })
        if requested is not None and requested - found:
            raise ValueError(f"Concepts absent from prompt CSV: {sorted(requested - found)}")
    else:
        contents = split_concepts(args.contents) if args.contents else (
            list(PAPER_CONTENTS[task]) if args.benchmark else list(targets)
        )
        if len({concept_key(x) for x in contents}) != len(contents):
            raise ValueError("Evaluation concepts contain duplicate names/aliases.")
        records = sample_records(
            contents, task, args.sample_template_count, args.num_samples,
            args.seed, args.sample_template_offset,
        )
    if not records:
        raise ValueError("The evaluation set is empty.")
    target_keys = {concept_key(t) for t in targets}
    present = {concept_key(r["concept"]) for r in records}
    if args.benchmark and target_keys - present:
        raise ValueError("Benchmark is missing an erasure target; supply --contents or --prompt-csv covering every target.")
    for row in records:
        row["is_target"] = concept_key(row["concept"]) in target_keys
    return records


def prepare_sample_plan(output, checkpoint, metadata, records, args):
    """A resume must match the checkpoint, prompt list and generation settings."""
    settings = {name: getattr(args, name) for name in (
        "architecture", "sd_ckpt", "flux_variant", "max_sequence_length", "dtype", "device",
        "steps", "guidance_scale", "true_cfg_scale", "negative_prompt", "height", "width",
    )}
    # The same CUDA backend can run on a different physical GPU when a batch
    # changes from one worker to two. GPU ordinals are routing, not a new dataset.
    def comparable(value):
        value = {**value, "sampling": dict(value["sampling"])}
        if re.fullmatch(r"cuda(?::[0-9]+)?", str(value["sampling"].get("device"))):
            value["sampling"]["device"] = "cuda"
        return value
    plan = {
        "schema": "casu2-evaluation-v1", "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": file_sha256(checkpoint / "weights.safetensors"),
        "checkpoint_metadata_sha256": file_sha256(checkpoint / "metadata.json"),
        "task": metadata["task"], "targets": metadata["targets"],
        "sampling": settings,
        "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "record_count": len(records),
    }
    plan = comparable(plan)
    path = output / "dataset.json"
    if args.resume and any(output.iterdir()):
        if not path.is_file():
            raise ValueError("Resume needs dataset.json from this sampler; use a new output directory.")
        previous = comparable(json.loads(path.read_text()))
        if previous != plan:
            differences = [key for key in plan if key != "sampling" and previous.get(key) != plan[key]]
            differences += [
                f"sampling.{key}: {previous['sampling'].get(key)!r} -> {value!r}"
                for key, value in plan["sampling"].items()
                if previous["sampling"].get(key) != value
            ]
            raise ValueError(
                "Resume settings, checkpoint or prompts differ from dataset.json: "
                + "; ".join(differences) + ". Use a new output directory for a changed experiment."
            )
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan
