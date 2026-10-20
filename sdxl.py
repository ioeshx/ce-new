"""SDXL base: concatenated CLIP K/V plus pooled-text columns, not size IDs."""

from conditioning import Entry, build_condition_updates
from editing import candidate_groups


def build_updates(pipe, pairs, retains, args):
    entries = []
    if args.condition_branches in {"both", "tokens"}:
        for group, specification in candidate_groups(pipe.unet, args.params).items():
            for key, module in specification["weights"].items():
                entries.append(Entry(group, key, module, "tokens", specification["parent"]))
    if args.condition_branches in {"both", "pooled"}:
        # The pooled vector precedes all size/crop time-ID features.
        pooled_dim = int(pipe.text_encoder_2.config.projection_dim)
        linear = pipe.unet.add_embedding.linear_1
        entries.append(Entry(
            "add_embedding.text", "add_embedding.linear_1.weight",
            linear, "pooled", text_columns=pooled_dim,
        ))
    return build_condition_updates(pipe, pipe.unet, "sdxl", entries, pairs, retains, args)
