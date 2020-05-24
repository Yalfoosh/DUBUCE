#   Copyright 2020 Miljenko Šuflaj
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Dict, Tuple

import numpy as np


def get_random_params(range_dict: Dict[str, Tuple] = None,
                      force_dict: Dict[str, Tuple] = None,
                      amount: int = 1):
    params_list = list()

    if range_dict is not None:
        for _ in range(amount):
            params = dict()

            for key, value in range_dict.items():
                if isinstance(value[0], int):
                    params[key] = np.random.randint(*value)
                elif isinstance(value[0], float):
                    params[key] = np.random.uniform(*value)

            params_list.append(params)

    if force_dict is not None:
        for key, value in force_dict.items():
            t_params_list = list()

            for subvalue in value:
                for params in params_list:
                    params[key] = subvalue
                    t_params_list.append(dict(params))

            params_list = list(t_params_list)

    return params_list
