from functools import singledispatch
from typing import Optional

from outlines.fsm.types import python_types_to_regex
from outlinesmlx.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def format(
    model,
    python_type,
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    regex_str = python_types_to_regex(python_type)
    return regex(model, regex_str, max_tokens, sampler)
