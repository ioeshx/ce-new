"""SD1.x prompt pairs, explicit CLIP token alignment, and retain text loading."""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from data.prompt_template import imagenet_templates, painting_templates


@dataclass(frozen=True)
class PromptPair:
    concept: str
    anchor: str
    target_prompt: str
    anchor_prompt: str
    template_index: int


def split_concepts(value: str, *, allow_empty: bool = False) -> list[str]:
    values = [x.strip() for x in value.split(",")]
    if not allow_empty and any(not x for x in values):
        raise ValueError("Concepts must be nonempty comma-separated names.")
    return values


def templates_for(task: str) -> list[str]:
    if task == "instance":
        return list(imagenet_templates)
    if task == "style":
        return list(painting_templates)
    raise ValueError(f"Unsupported task: {task}")


def choose_templates(task: str, count: int, offset: int = 0) -> list[tuple[int, str]]:
    templates = templates_for(task)
    if count < 0 or not 0 <= offset < len(templates):
        raise ValueError("Template count must be >= 0 and offset must be in range.")
    end = len(templates) if count == 0 else min(offset + count, len(templates))
    return list(enumerate(templates))[offset:end]


def build_pairs(
    targets: list[str],
    anchors: list[str],
    task: str,
    contextual: bool,
    count: int,
) -> list[PromptPair]:
    if len(anchors) == 1:
        anchors = anchors * len(targets)
    if len(anchors) != len(targets):
        raise ValueError("Supply one anchor or one anchor per target.")
    if any(t.casefold() == a.casefold() for t, a in zip(targets, anchors)):
        raise ValueError("A target and its anchor cannot be identical.")
    pairs = []
    for target, anchor in zip(targets, anchors):
        if contextual:
            for index, template in choose_templates(task, count):
                # An empty anchor removes the placeholder, NOT the context.
                reference = re.sub(r"\s+", " ", template.format(anchor)).strip()
                pairs.append(PromptPair(target, anchor, template.format(target), reference, index))
        else:
            pairs.append(PromptPair(target, anchor, target, anchor, -1))
    return pairs


def pair_records(pairs: list[PromptPair]) -> list[dict]:
    return [asdict(pair) for pair in pairs]


def load_retain_texts(
    path: Path | None,
    column: str,
    targets: list[str],
    count: int,
) -> list[str]:
    if path is None or count == 0:
        return []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if column not in (reader.fieldnames or []):
                raise ValueError(f"{path}: no column {column!r}; columns={reader.fieldnames}")
            texts = [row[column].strip() for row in reader if row.get(column, "").strip()]
    else:
        texts = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    targets = [t.casefold() for t in targets]
    result, seen = [], set()
    for text in texts:
        key = text.casefold()
        # Also exclude a shorter retain alias such as Mickey for Mickey Mouse.
        overlaps = any(
            re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", key)
            or re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", t)
            for t in targets
        )
        if key not in seen and not overlaps:
            seen.add(key)
            result.append(text)
    return result if count < 0 else result[:count]


def aligned_positions(
    target_ids: list[int],
    anchor_ids: list[int],
    target_length: int,
    anchor_length: int,
    mode: str = "full",
    include_padding: bool = True,
    has_bos: bool = True,
    has_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Align actual token positions, never average hidden states.

    Equal spans match by token ID; a changed span uses endpoint-preserving
    nearest-neighbour resampling. A single source token maps to the LAST
    replacement token, which has seen the whole replacement in causal CLIP.
    EOS and padding are matched by their offset from EOS. Chat-formatted
    sequences can omit both BOS and terminal EOS via the corresponding flags.
    """
    start = 1 if has_bos else 0
    if min(target_length, anchor_length) < start + 1:
        raise ValueError("Tokenized prompts must contain at least one non-BOS token.")
    source = target_ids[start:target_length - int(has_eos)]
    reference = anchor_ids[start:anchor_length - int(has_eos)]
    matcher = difflib.SequenceMatcher(a=source, b=reference, autojunk=False)
    ti, ai = [], []
    for tag, i0, i1, j0, j1 in matcher.get_opcodes():
        if tag == "insert" or (mode == "subject" and tag == "equal"):
            continue
        if tag == "equal":
            matches = list(range(j0, j1))
        elif j0 == j1:
            # Removed concept: use the next available token, or EOS.
            last = len(reference) if has_eos else max(0, len(reference) - 1)
            matches = [min(j0, last)] * (i1 - i0)
        elif i1 - i0 == 1:
            matches = [j1 - 1]
        else:
            matches = [
                j0 + round(k * (j1 - j0 - 1) / (i1 - i0 - 1))
                for k in range(i1 - i0)
            ]
        ti.extend(range(i0 + start, i1 + start))
        ai.extend(j + start for j in matches)
    if mode == "full":
        if has_eos:
            ti.append(target_length - 1)
            ai.append(anchor_length - 1)
        if include_padding:
            for pos in range(target_length, len(target_ids)):
                ti.append(pos)
                ai.append(min(anchor_length + pos - target_length, len(anchor_ids) - 1))
    if not ti:
        raise ValueError("The prompt pair has no editable aligned token positions.")
    return ti, ai
