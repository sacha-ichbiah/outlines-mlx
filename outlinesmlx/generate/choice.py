from functools import singledispatch
from typing import Callable, List, Optional

from outlinesmlx.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def choice(
    model,
    choices: List[str],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    regex_str = r"(" + r"|".join(choices) + r")"
    return regex(model, regex_str, max_tokens, sampler)

