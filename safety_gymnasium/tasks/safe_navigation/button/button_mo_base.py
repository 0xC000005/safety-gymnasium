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
"""Button task 0."""

import gymnasium
import mujoco
import numpy as np

from safety_gymnasium.bases.base_task import BaseTask


# pylint: disable-next=too-many-instance-attributes
class ButtonMOBase(BaseTask):
    """An agent must press a goal button."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.last_dist_goal = None

    def calculate_reward(self):
        """
        The new multi-objective reward function is:
        r_time = D_last - D_now, where D is the distance to the goal button
        r_energy = -|a|^2, where a is the acceleration of the agent
        r_success = 100 if the current button is pressed, 0 otherwise

        Returns:
            np.ndarray: vector_reward = np.array([r_time, r_energy, r_success])
        """
        # r_time: progress towards the goal
        dist_goal = self.dist_goal()
        r_time = self.last_dist_goal - dist_goal if self.last_dist_goal is not None else 0.0
        self.last_dist_goal = dist_goal

        # r_energy: negative squared norm of the agent's acceleration
        acceleration = self.agent.acc  # Use the new acc property
        r_energy = -np.sum(np.square(acceleration))

        # r_success: large reward for pressing the correct button
        r_success = 100.0 if self.goal_achieved else 0.0

        vector_reward = np.array([r_time, r_energy, r_success], dtype=np.float32)
        return vector_reward

    def specific_reset(self):
        """Reset the buttons timer."""
        self.buttons.timer = 0  # pylint: disable=no-member

    def specific_step(self):
        """Clock the buttons timer."""
        self.buttons.timer_tick()  # pylint: disable=no-member

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        assert self.buttons.num > 0, 'Must have at least one button.'
        self.build_goal_button()
        self.last_dist_goal = self.dist_goal()
        self.buttons.reset_timer()  # pylint: disable=no-member

    def build_goal_button(self):
        """Pick a new goal button, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        self.buttons.goal_button = self.random_generator.choice(self.buttons.num)
        new_goal_pos = self.buttons.pos[self.buttons.goal_button]  # pylint: disable=no-member
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = new_goal_pos[:2]
        self._set_goal(new_goal_pos[:2])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos, obstacle.group)
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        if self.buttons.timer != 0:  # pylint: disable=no-member
            obs['buttons_lidar'] = np.zeros(self.lidar_conf.num_bins)

        if self.observe_vision:
            obs['vision'] = self._obs_vision()

        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'

        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)
        return obs

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            # pylint: disable-next=no-member
            if any(n == f'button{self.buttons.goal_button}' for n in geom_names) and any(
                n in self.agent.body_info.geom_names for n in geom_names
            ):
                return True
        return False

