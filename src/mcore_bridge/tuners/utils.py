# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift

from typing import Optional, Tuple

from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default


def tuners_sharded_state_dict(
        module,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
):
    sharded_state_dict = {}
    # Save parameters
    module._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
    sharded_state_dict = make_sharded_tensors_for_checkpoint(
        sharded_state_dict, prefix, sharded_offsets=sharded_offsets)
    # Recurse into submodules
    for name, child in module.named_children():
        if 'Dict' in child.__class__.__name__:
            modules = child.named_children()
        else:
            modules = [(None, child)]
        for n, m in modules:
            _prefix = f'{prefix}{name}.' if n is None else f'{prefix}{name}.{n}.'
            sharded_state_dict.update(sharded_state_dict_default(m, _prefix, sharded_offsets, metadata))
    return sharded_state_dict
