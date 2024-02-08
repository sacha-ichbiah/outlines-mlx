import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Union

import mlx.core as mx
import numpy as np 
from outlines.fsm.fsm import FSMState

if TYPE_CHECKING:
    from outlines.fsm.fsm import FSM
    from outlinesmlx.samplers import Sampler


@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: mx.array
    kv_cache: mx.array
    logits: mx.array
    fsm_states: List[FSMState]


def sequence_generator(
    token_generator: Callable,
    fsms: List["FSM"],
    token_ids: mx.array,
    attention_masks: mx.array,
    fsm_states: List[FSMState],
) -> Iterator[GenerationState]:
    """Generates sequences of tokens.

    Parameters
    ----------
    token_generator
        A callable that generate a new token given the current generation state
        and logits biases.
    fsms
        List of finite-state machines that drive the text generation,
        one for each sequence in the batch.
    init_state
        The initial generation state for the batches.
    fsm_states
        The initial states of the finite-state machine for each sequence in the batch.

    Yields
    ------
    A new sequence.

    """
    kv_cache = None

    while True:
        allowed_tokens = get_allowed_tokens(fsms, fsm_states)

        next_token_ids, kv_cache, logits, _ = token_generator(
            token_ids,
            attention_masks,
            kv_cache,
            allowed_tokens=allowed_tokens,
        )
        token_ids = update_token_ids(token_ids, next_token_ids)
        attention_masks = expand_attention_masks(attention_masks)

        fsm_states = get_next_fsm_states(fsms, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsms, fsm_states)

        if is_finished:
            yield GenerationState(
                token_ids,
                kv_cache,
                logits,
                fsm_states,
            )
            return

        yield GenerationState(
            token_ids,
            kv_cache,
            logits,
            fsm_states,
        )


def token_generator(model, sampler: "Sampler") -> Callable:
    """Generate one token at a time.

    This process is designed to be steered by another supervising
    process that supplies the current sequence and the indices
    of the tokens to mask before sampling.

    Parameters
    ----------
    model
        A model that takes a sequence of tokens as an input and
        returns a probability distribution over the next tokens.
    sampler
        A function that samples tokens from a probability
        distribution over the next tokens.

    Returns
    -------
    A tuple that contains a tensor with the sampled tokens, a tensor with
    the K-V cache for the sequence and the tensor that contains the next-token
    logits that were returned by the model.

    """

    def generate(
        token_ids: mx.array,
        attention_masks: mx.array,
        kv_cache: mx.array,
        allowed_tokens: List[List[int]],
    ) -> Union[mx.array, mx.array, mx.array, mx.array]:
        try:
            logits, new_kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:  # Exceeding the context length
            raise IndexError(
                "The input length exceeds the context length of the model."
            )

        biased_logits = bias_logits(logits, allowed_tokens)
        next_token_ids = sampler(biased_logits)

        return next_token_ids, new_kv_cache, logits, biased_logits

    return generate


def get_next_fsm_states(
    fsms: List["FSM"], fsm_states: List[FSMState], next_token_ids: mx.array
) -> List[FSMState]:
    """

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    next_token_ids
        The tokens that were just generated.

    Returns
    -------
    A `mx.array` object that represents the next logit mask.

    """

    return [
        fsm.next_state(fsm_state, int(token_id[0].item()))
        for fsm, fsm_state, token_id in zip(fsms, fsm_states, next_token_ids)
    ]


def get_allowed_tokens(fsms: List["FSM"], fsm_states: List[FSMState]) -> List[List[int]]:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to keep.

    """
    return [fsm.allowed_token_ids(state) for fsm, state in zip(fsms, fsm_states)]


def is_generation_finished(fsms: List["FSM"], fsm_states: List[FSMState]) -> bool:
    """Determine if the generation is finished.

    A generation is considered finished if the FSM of every sequence in the
    batch is in a final state.

    A better solution is to return finished sequences as soon as their FSM
    is in a final state.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    Whether all sequences are finished sampling.

    """
    return all([fsm.is_final_state(state) for fsm, state in zip(fsms, fsm_states)])


def update_token_ids(
    token_ids: mx.array, next_token_ids: mx.array
) -> mx.array:
    """Append the sampled tokens to the running sequence of tokens.

    Parameters
    ----------
    token_ids
        The current token sequences
    next_token_ids
        The tokens that were just generated and that we need to append
        to the existing sequences.

    Returns
    -------
    A new sequence of token ids that contains the tokens that were
    just generated.

    """
    return mx.concatenate([token_ids, next_token_ids], axis=-1)


def expand_attention_masks(attention_masks: mx.array) -> mx.array:
    """Expand the attention masks.

    Parameters
    ----------
    attention_masks
        The attention masks for each sequence in the batch.

    Returns
    -------
    The attention masks padded with 1s.

    """
    return mx.concatenate(
        [
            attention_masks,
            mx.ones(
                attention_masks.shape[:-1] + (1,)
            ),
        ],
        axis=-1,
    )


def bias_logits(logits: mx.array, allowed_token_ids: List) -> mx.array:
    """Mask the logits.

    The function iterates over a nested list where each list corresponds to the
    indices that need to be masked for each row in the array.

    Parameters
    ----------
    logits
        Two dimensional tensor that contains the next-token probability
        distribution.
    allowed_token_ids
        A list that contains the tokens that can be generated by the model.

    Returns
    -------
    A view of the original logits tensor where some values are masked.

    """
    biased_logits = mx.full(logits.shape, -math.inf)
    for i, ids in enumerate(allowed_token_ids):
        idx = mx.array(ids)
        biased_logits[i, idx] = logits[i, idx]
        
    return biased_logits

