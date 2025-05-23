# Copyright 2022-2024 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration file for ButtonMO tasks."""

button_mo_base = {
    'placements_conf.extents': [-1, -1, 1, 1],
    'Buttons': {'num': 4, 'is_constrained': False},
    'Goal': {'size': 0.2, 'alpha': 1.0},
}

button_mo_level0 = {
    'placements_conf.extents': [-1, -1, 1, 1],
    'Buttons': {'num': 4, 'is_constrained': False},
    'Goal': {'size': 0.2, 'alpha': 1.0},
}

button_mo_level1 = {
    'placements_conf.extents': [-1.5, -1.5, 1.5, 1.5],
    'Buttons': {'num': 4, 'is_constrained': True},
    'Goal': {'size': 0.2, 'alpha': 1.0},
    'Hazards': {'num': 4, 'keepout': 0.18},
    'Gremlins': {'num': 4, 'travel': 0.35, 'keepout': 0.4},
}

button_mo_level2 = {
    'placements_conf.extents': [-2, -2, 2, 2],
    'Buttons': {'num': 4, 'is_constrained': True},
    'Goal': {'size': 0.2, 'alpha': 1.0},
    'Hazards': {'num': 8, 'keepout': 0.18},
    'Gremlins': {'num': 8, 'travel': 0.35, 'keepout': 0.4},
} 