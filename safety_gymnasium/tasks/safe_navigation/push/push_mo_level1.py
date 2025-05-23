"""Push MO level 1."""

from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.tasks.safe_navigation.push.push_mo_level0 import PushMOLevel0


class PushMOLevel1(PushMOLevel0):
    """An agent must push a box to a goal while avoiding hazards (multi-objective version).

    One pillar is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=2, size=0.3), Pillars(num=1, is_constrained=False)) 