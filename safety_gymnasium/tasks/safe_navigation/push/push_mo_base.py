"""Push with a custom config."""

import numpy as np

from safety_gymnasium.bases.base_task import BaseTask


class PushMOBase(BaseTask):
    """An agent must push a box to a goal."""

    def __init__(self, config) -> None:
        assert 'Goal' in config, '`config` must have the field `Goal`'
        assert 'PushBox' in config, '`config` must have the field `PushBox`'
        super().__init__(config=config)

        self.last_dist_box = None
        self.last_box_goal = None
        self.last_dist_goal = None

    def calculate_reward(self):
        """
        The multi-objective reward function for push task:
        r_time = D_last - D_now, where D is the distance from box to goal
        r_energy = -|a|^2, where a is the acceleration of the agent
        r_success = 100 if the box reaches the goal, 0 otherwise

        Returns:
            np.ndarray: vector_reward = np.array([r_time, r_energy, r_success])
        """
        # r_time: progress of the box towards the goal
        dist_box_goal = self.dist_box_goal()
        r_time = self.last_box_goal - dist_box_goal if self.last_box_goal is not None else 0.0
        self.last_box_goal = dist_box_goal

        # r_energy: negative squared norm of the agent's acceleration
        acceleration = self.agent.acc  # Use the new acc property
        r_energy = -np.sum(np.square(acceleration))

        # r_success: large reward for successfully pushing box to goal
        r_success = 100.0 if self.goal_achieved else 0.0

        vector_reward = np.array([r_time, r_energy, r_success], dtype=np.float32)
        return vector_reward

        
    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()

    def dist_box(self):
        """Return the distance. from the agent to the box (in XY plane only)"""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.agent.pos)))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position."""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.goal.pos)))

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_box_goal() <= self.goal.size
