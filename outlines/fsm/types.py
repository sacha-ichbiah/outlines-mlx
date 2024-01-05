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

import datetime
from typing import Any

INTEGER = r"[+-]?(0|[1-9][0-9]*)"
BOOLEAN = "(True|False)"
FLOAT = rf"{INTEGER}(\.[0-9]+)?([eE][+-][0-9]+)?"
DATE = r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"
TIME = r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
DATETIME = rf"({DATE})(\s)({TIME})"


def python_types_to_regex(python_type: Any) -> str:
    if python_type == float:
        return FLOAT
    elif python_type == int:
        return INTEGER
    elif python_type == bool:
        return BOOLEAN
    elif python_type == datetime.date:
        return DATE
    elif python_type == datetime.time:
        return TIME
    elif python_type == datetime.datetime:
        return DATETIME
    else:
        raise NotImplementedError(
            f"The Python type {python_type} is not supported. Please open an issue."
        )
