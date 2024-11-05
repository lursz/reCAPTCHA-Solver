import pyautogui
import numpy as np
import time

from .mouseContext import MouseContext
from .movementStategies.base import MovementStrategy
from .movementStategies.vectoredStrategy import MoveAllTheWayStrategy


class MouseEngine:
    def __init__(self) -> None:
        self.context = MouseContext()
        self.strategy: MovementStrategy = None

    def set_strategy(self, strategy: MovementStrategy):
        self.strategy = strategy

    def move_mouse(self):
        if self.strategy:
            self.strategy.move(self.context)
        else:
            raise ValueError("Movement strategy not set")

    def move_mouse_all_the_way(self, position) -> None:
        self.context.target = position
        self.set_strategy(MoveAllTheWayStrategy())
        self.move_mouse()