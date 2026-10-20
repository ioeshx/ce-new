"""FLUX.2-klein-4B: edit only the Qwen3 text-sequence entry."""

from conditioning import Entry, build_condition_updates


def build_updates(pipe, pairs, retains, args):
    model = pipe.transformer
    if args.condition_branches == "pooled":
        raise ValueError("FLUX.2-klein has no pooled-text injection branch.")
    entries = []
    if args.condition_branches in {"both", "tokens"}:
        entries.append(Entry(
            "context_embedder", "context_embedder.weight", model.context_embedder, "tokens",
        ))
    # No shared single-stream weights are edited, so image token maps stay intact.
    return build_condition_updates(pipe, model, "flux", entries, pairs, retains, args)
