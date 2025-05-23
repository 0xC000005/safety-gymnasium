from safety_gymnasium.tasks.safe_navigation.button.button_mo_base import ButtonMOBase as ButtonMOBase
from safety_gymnasium.assets.geoms import Buttons, Goal
from safety_gymnasium.utils.registration import register

class ButtonMOLevel0(ButtonMOBase):
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1, -1, 1, 1]
        self._add_geoms(Buttons(num=4, is_constrained=False))
        self._add_geoms(Goal(size=self.buttons.size * 2, alpha=1.0))
        self.last_dist_goal = None
