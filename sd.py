"""SD1.x: edit text cross-attention K/V with the shared conditioning solver."""

from conditioning import Entry, build_condition_updates
from editing import candidate_groups


def build_updates(pipe, pairs, retains, args):
    entries = []
    for group, specification in candidate_groups(pipe.unet, args.params).items():
        for key, module in specification["weights"].items():
            entries.append(Entry(group, key, module, "tokens", specification["parent"]))
    return build_condition_updates(pipe, pipe.unet, "sd1", entries, pairs, retains, args)
