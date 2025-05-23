# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Wrappers for converting between Safety-Gymnasium and Gymnasium environments."""


import gymnasium
from gymnasium import logger
from gymnasium.core import ActType


class SafetyGymnasium2Gymnasium(gymnasium.Wrapper):
    """A class that converts a Safety-Gymnasium environment to a Gymnasium environment."""

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        if 'cost' in info:
            logger.warn(
                'The info dict already contains a cost. '
                'Overwriting it may cause unexpected behavior.',
            )
        info['cost'] = cost
        return obs, reward, terminated, truncated, info


class Gymnasium2SafetyGymnasium(gymnasium.Wrapper):
    """A class that converts a Gymnasium environment to a Safety-Gymnasium environment."""

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = super().step(action)
        try:
            cost = info['cost']
        except KeyError as ex:
            raise ValueError(
                'The info dict does not contain a cost which is required by Safety-Gymnasium.',
            ) from ex
        return obs, reward, cost, terminated, truncated, info


class SafetyMOGymnasium2SafetyGymnasium(gymnasium.Wrapper):
    """A wrapper that combines 'r_time' and 'r_success' from a reward dict or vector into a scalar reward, ignoring other components."""
    def step(self, action):
        result = self.env.step(action)
        
        # Handle both Safety-Gymnasium format (6 return values) and Gymnasium format (5 return values)
        if len(result) == 6:
            obs, vector_reward, cost, terminated, truncated, info = result
        else:
            obs, vector_reward, terminated, truncated, info = result
            cost = info.get('cost', 0.0)
        
        # Convert vector reward to scalar
        if isinstance(vector_reward, dict):
            reward = float(vector_reward.get('r_time', 0.0)) + float(vector_reward.get('r_success', 0.0))
        else:
            # Assume vector_reward is a numpy array: [r_time, r_energy, r_success]
            reward = float(vector_reward[0]) + float(vector_reward[2])
        
        # Return in Safety-Gymnasium format
        return obs, reward, cost, terminated, truncated, info


def make_gymnasium_environment(env_id, *args, **kwargs):
    """Make a Gymnasium environment."""

    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from safety_gymnasium.utils.registration import make

    env_name, _, version = env_id.partition('-')
    if not env_name.endswith('Gymnasium'):
        raise ValueError(f'Environment {env_id} is not a Gymnasium environment.')
    env_name = env_name[: -len('Gymnasium')]
    safe_env = make(f'{env_name}-{version}', *args, **kwargs)
    return SafetyGymnasium2Gymnasium(safe_env)
