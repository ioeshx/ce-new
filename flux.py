"""FLUX.2-klein: Qwen3 sequence entry; legacy FLUX.1: T5 + CLIP."""

from conditioning import Entry, build_condition_updates, is_klein


def build_updates(pipe, pairs, retains, args):
    model = pipe.transformer
    klein = is_klein(args)
    if klein and args.condition_branches == "pooled":
        raise ValueError("FLUX.2-klein has no pooled-text injection branch.")
    entries = []
    if args.condition_branches in {"both", "tokens"}:
        entries.append(Entry(
            "context_embedder", "context_embedder.weight", model.context_embedder, "tokens",
        ))
    if not klein and args.condition_branches in {"both", "pooled"}:
        entries.append(Entry(
            "pooled_text", "time_text_embed.text_embedder.linear_1.weight",
            model.time_text_embed.text_embedder.linear_1, "pooled",
        ))
    # No shared single-stream weights are edited, so image token maps stay intact.
    return build_condition_updates(pipe, model, "flux", entries, pairs, retains, args)
