"""
Copyright 2023- The Outlines developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


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
