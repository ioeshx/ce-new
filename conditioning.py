"""Actual conditioning inputs for SDXL, SD3.0 and FLUX.2/FLUX.1."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from prompts import aligned_positions
from models import is_klein
from sd import EditController, actual_kv_input
from solver import solve_factors


@dataclass
class Segment:
    name: str
    offset: int
    ids: list[int]
    length: int
    has_bos: bool
    has_eos: bool = True
    retain_index: int | None = None


@dataclass
class Condition:
    tokens: torch.Tensor
    pooled: torch.Tensor | None
    segments: list[Segment]


@dataclass
class Entry:
    group: str
    key: str
    module: torch.nn.Linear
    kind: str
    attention: object = None
    text_columns: int | None = None


def execution_device(pipe):
    return pipe._execution_device


class ConditionCache:
    """Use the pipeline's own encoder/layer/concatenation convention."""

    def __init__(self, pipe, architecture, args):
        self.pipe, self.architecture, self.args = pipe, architecture, args
        self.cache = {}

    def tokenize(self, tokenizer, text, length, name, offset=0, has_bos=True):
        raw = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        if len(raw) > length:
            raise ValueError(f"{name} prompt exceeds {length} tokens: {text!r}")
        encoded = tokenizer(
            text, padding="max_length", max_length=length, truncation=True,
            add_special_tokens=True, return_tensors="pt",
        )
        return Segment(name, offset, encoded.input_ids[0].tolist(),
                       int(encoded.attention_mask[0].sum()), has_bos)

    def segments(self, text):
        p, arch = self.pipe, self.architecture
        if is_klein(self.args):
            return [self.klein_segment(text)]
        clip_length = int(p.tokenizer.model_max_length)
        clip = self.tokenize(p.tokenizer, text, clip_length, "clip")
        if arch in {"sdxl", "sd3"}:
            other = self.tokenize(p.tokenizer_2, text, clip_length, "clip_2")
            # Standard SDXL/SD3 CLIP encoders may use DIFFERENT pad token IDs.
            # Only valid tokens need shared positions; keep each encoder's
            # native padding states in the actual concatenated representation.
            if clip.length != other.length or clip.ids[:clip.length] != other.ids[:other.length]:
                raise ValueError(
                    "The concatenated CLIP encoders tokenize differently; this adapter "
                    "requires the standard shared CLIP tokenization."
                )
        if arch == "sdxl":
            return [clip]
        tokenizer = p.tokenizer_3 if arch == "sd3" else p.tokenizer_2
        if tokenizer is None:
            raise ValueError("The T5 text branch is required; do not omit its tokenizer/encoder.")
        t5 = self.tokenize(
            tokenizer, text, self.args.max_sequence_length, "t5",
            offset=clip_length if arch == "sd3" else 0, has_bos=False,
        )
        return [clip, t5] if arch == "sd3" else [t5]

    def klein_segment(self, text):
        """Match the native Qwen3 chat formatting, including assistant suffix.

        Retain the last token of the user's content, rather than a chat-control
        token after that content. Offsets are from the same fast tokenizer.
        """
        tokenizer = self.pipe.tokenizer
        if tokenizer.padding_side != "right":
            raise ValueError("Klein token alignment requires right padding, as in the standard tokenizer.")

        def render(content):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )

        formatted = render(text)
        length = self.args.max_sequence_length
        raw = tokenizer(formatted, truncation=False)["input_ids"]
        if len(raw) > length:
            raise ValueError(f"Qwen3 chat-formatted prompt exceeds {length} tokens: {text!r}")
        encoded = tokenizer(
            formatted, padding="max_length", max_length=length, truncation=True,
            return_tensors="pt", return_offsets_mapping=True,
        )
        marker = "__CASU_USER_CONTENT_BOUNDARY__"
        template = render(marker)
        if template.count(marker) != 1:
            raise ValueError("Unsupported Klein chat template: cannot locate user content.")
        prefix, suffix = template.split(marker)
        if formatted != prefix + text + suffix:
            raise ValueError("Unsupported Klein chat template: user content is transformed.")
        positions = [
            i for i, (start, end) in enumerate(encoded.offset_mapping[0].tolist())
            if end > start and start < len(prefix) + len(text) and end > len(prefix)
        ]
        valid_length = int(encoded.attention_mask[0].sum())
        return Segment(
            "qwen3", 0, encoded.input_ids[0].tolist(), valid_length, False,
            has_eos=False, retain_index=positions[-1] if positions else valid_length - 1,
        )

    @torch.inference_mode()
    def preload(self, texts):
        pending = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for start in range(0, len(pending), self.args.encode_batch_size):
            chunk = pending[start:start + self.args.encode_batch_size]
            segments = [self.segments(text) for text in chunk]
            kwargs = dict(prompt=chunk, device=execution_device(self.pipe), num_images_per_prompt=1)
            klein = is_klein(self.args)
            if self.architecture != "flux":
                kwargs["do_classifier_free_guidance"] = False
            if self.architecture == "sd3":
                kwargs.update(prompt_2=None, prompt_3=None, max_sequence_length=self.args.max_sequence_length)
            elif self.architecture == "flux":
                kwargs["max_sequence_length"] = self.args.max_sequence_length
            result = self.pipe.encode_prompt(**kwargs)
            tokens = result[0]
            pooled = None if klein else result[1] if self.architecture == "flux" else result[2]
            for i, text in enumerate(chunk):
                expected = sum(len(s.ids) for s in segments[i])
                if tokens.shape[1] != expected:
                    raise ValueError("Pipeline token layout differs from the adapter's segment layout.")
                self.cache[text] = Condition(
                    tokens[i].detach().float().cpu(),
                    pooled[i].detach().float().cpu() if pooled is not None else None, segments[i],
                )
            print(f"[encode:{self.architecture}] {min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)

    def get(self, text):
        if text not in self.cache:
            self.preload([text])
        return self.cache[text]


def capped_alignment(target, anchor, args):
    """Select actual rows; keep changed span, EOS and a tail-padding sample."""
    ti, ai = aligned_positions(
        target.ids, anchor.ids, target.length, anchor.length,
        args.token_mode, args.include_padding, has_bos=target.has_bos, has_eos=target.has_eos,
    )
    budget = args.max_fit_tokens
    if budget and len(ti) > budget:
        try:
            changed, _ = aligned_positions(
                target.ids, anchor.ids, target.length, anchor.length,
                "subject", False, has_bos=target.has_bos, has_eos=target.has_eos,
            )
        except ValueError:
            changed = []
        required = set(changed) | {target.length - 1}
        if args.include_padding:
            required.add(len(target.ids) - 1)
        selected = {i for i, pos in enumerate(ti) if pos in required}
        remaining = [i for i in range(len(ti)) if i not in selected]
        slots = max(0, budget - len(selected))
        if slots:
            picks = torch.linspace(0, len(remaining) - 1, min(slots, len(remaining))).round().int().tolist()
            selected.update(remaining[i] for i in picks)
        indices = sorted(selected)
        ti, ai = [ti[i] for i in indices], [ai[i] for i in indices]
    return [i + target.offset for i in ti], [i + anchor.offset for i in ai]


def input_matrices(cache, pairs, retains, entry, args):
    source, anchor, protection = [], [], []

    def states(condition):
        return actual_kv_input(entry.attention, condition.tokens) if entry.attention is not None else condition.tokens

    for pair in pairs:
        target, reference = cache.get(pair.target_prompt), cache.get(pair.anchor_prompt)
        if entry.kind == "pooled":
            source.append(target.pooled[None])
            anchor.append(reference.pooled[None])
        else:
            th, ah = states(target), states(reference)
            for ts, ass in zip(target.segments, reference.segments):
                ti, ai = capped_alignment(ts, ass, args)
                source.append(th[ti])
                anchor.append(ah[ai])
    for text in retains:
        condition = cache.get(text)
        if entry.kind == "pooled":
            protection.append(condition.pooled[None])
        else:
            indices = [
                s.offset + (s.retain_index if s.retain_index is not None else max(int(s.has_bos), s.length - 2))
                for s in condition.segments
            ]
            protection.append(states(condition)[indices])
    if args.protect_null:
        condition = cache.get("")
        protection.append(condition.pooled[None] if entry.kind == "pooled" else states(condition))
    xt, xa = torch.cat(source), torch.cat(anchor)
    xp = torch.cat(protection) if protection else xt.new_empty((0, xt.shape[1]))
    return xt, xa, xp


@torch.no_grad()
def build_condition_updates(pipe, backbone, architecture, entries, pairs, retains, args):
    cache = ConditionCache(pipe, architecture, args)
    texts = [x for p in pairs for x in (p.target_prompt, p.anchor_prompt)] + list(retains)
    if args.protect_null:
        texts.append("")
    cache.preload(texts)
    factors, original, edited, groups, diagnostics = {}, {}, {}, {}, {}
    for entry in entries:
        if not isinstance(entry.module, torch.nn.Linear):
            raise ValueError(f"{entry.key}: only unfused torch.nn.Linear projections are supported.")
        normalized = entry.attention is not None and getattr(entry.attention, "norm_cross", False)
        input_key = (entry.kind, entry.group if normalized else "plain")
        if input_key not in factors:
            xt, xa, xp = input_matrices(cache, pairs, retains, entry, args)
            factors[input_key] = solve_factors(
                xt, xa, xp, beta=args.beta, rank_rtol=args.rank_rtol,
                preserve_rank=args.preserve_rank, device=args.solve_device,
            )
        left, right, info = factors[input_key]
        w = entry.module.weight.detach().cpu().clone()
        columns = entry.text_columns if entry.text_columns is not None else w.shape[1]
        if columns != info["input_dim"]:
            raise ValueError(f"{entry.key}: actual input width does not match editable columns.")
        local = w[:, :columns].to(device=args.solve_device, dtype=torch.float64)
        update = (local @ left.to(args.solve_device)) @ right.to(args.solve_device)
        value = w.clone()
        value[:, :columns] = (local + update).to(device="cpu", dtype=w.dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite edited weight: {entry.key}")
        original[entry.key], edited[entry.key] = w, value
        groups.setdefault(entry.group, []).append(entry.key)
        diagnostics[entry.group] = {
            **info, "input_kind": entry.kind, "editable_columns": columns,
            "actual_input": "post_cross_norm" if normalized else entry.kind,
        }
        print(f"[solve:{architecture}] {entry.key}: rank={info['protected_rank']}, null={info['null_dim']}", flush=True)
    if getattr(pipe, "_casu_cpu_offload", False):
        pipe.maybe_free_model_hooks()
    return EditController(backbone, original, edited, groups, architecture), diagnostics
