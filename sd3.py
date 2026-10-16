"""SD3.0: the text-sequence entry and pooled-CLIP entry into MMDiT."""

from conditioning import Entry, build_condition_updates


def build_updates(pipe, pairs, retains, args):
    model = pipe.transformer
    entries = []
    if args.condition_branches in {"both", "tokens"}:
        entries.append(Entry(
            "context_embedder", "context_embedder.weight", model.context_embedder, "tokens",
        ))
    if args.condition_branches in {"both", "pooled"}:
        entries.append(Entry(
            "pooled_text", "time_text_embed.text_embedder.linear_1.weight",
            model.time_text_embed.text_embedder.linear_1, "pooled",
        ))
    return build_condition_updates(pipe, model, "sd3", entries, pairs, retains, args)
