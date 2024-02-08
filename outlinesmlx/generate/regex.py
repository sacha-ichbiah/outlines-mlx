from functools import singledispatch
from typing import Optional

from outlines.fsm.fsm import RegexFSM
from outlinesmlx.generate.api import SequenceGenerator
from outlinesmlx.samplers import Sampler, multinomial


@singledispatch
def regex(
    model,
    regex_str: str,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    fsm = RegexFSM(regex_str, model.tokenizer)

    generator = SequenceGenerator(fsm, model, sampler, max_tokens=max_tokens)

    return generator

