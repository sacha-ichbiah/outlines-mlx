import warnings
from functools import singledispatch
from typing import List, Optional, Union

from outlines.fsm.fsm import StopAtEosFSM
from outlinesmlx.generate import SequenceGenerator
from outlinesmlx.samplers import Sampler, multinomial


@singledispatch
def text(
    model,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[str, List[str]]] = None,
    *,
    samples: int = 1,
    sampler: Sampler = multinomial(),
) -> SequenceGenerator:
    """Generate text with a `Transformer` model.

    Note
    ----
    Python 3.11 allows dispatching on Union types and
    this should greatly simplify the code.

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    max_tokens:
        The maximum number of tokens to generate.
    stop_at:
        Text sequences such that the generation stops after they've been
        generated.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text.

    """
    if samples > 1:
        raise NotImplementedError(
            "It is currently impossible to generate several samples with `transformers` models."
        )

    fsm = StopAtEosFSM(model.tokenizer)

    generator = SequenceGenerator(
        fsm, model, sampler, max_tokens=max_tokens, stop_at=stop_at
    )

    return generator

