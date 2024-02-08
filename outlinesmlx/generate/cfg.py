from functools import singledispatch
from typing import List, Optional, Union

from outlines.fsm.fsm import CFGFSM
from outlinesmlx.generate.api import SequenceGenerator
from outlinesmlx.samplers import Sampler, multinomial


@singledispatch
def cfg(
    model,
    cfg_str: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    sampler: Sampler = multinomial(),
):
    fsm = CFGFSM(cfg_str, model.tokenizer)

    generator = SequenceGenerator(
        fsm, model, sampler,  max_tokens=max_tokens, stop_at=stop_at
    )

    return generator

