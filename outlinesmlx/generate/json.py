import json as pyjson
from functools import singledispatch
from typing import Callable, Optional, Union

from pydantic import BaseModel

from outlines.fsm.json_schema import build_regex_from_object, get_schema_from_signature
from outlinesmlx.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def json(
    model,
    schema_object: Union[str, object, Callable],
    max_tokens: Optional[int] = None,
    sampler: Sampler = multinomial(),
):
    if isinstance(schema_object, type(BaseModel)):
        schema = pyjson.dumps(schema_object.model_json_schema())
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: schema_object.parse_raw(x)
    elif callable(schema_object):
        schema = pyjson.dumps(get_schema_from_signature(schema_object))
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    elif isinstance(schema_object, str):
        schema = schema_object
        regex_str = build_regex_from_object(schema)
        generator = regex(model, regex_str, max_tokens, sampler)
        generator.format_sequence = lambda x: pyjson.loads(x)
    else:
        raise ValueError(
            f"Cannot parse schema {schema_object}. The schema must be either "
            + "a Pydantic object, a function or a string that contains the JSON "
            + "Schema specification"
        )

    return generator

