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

