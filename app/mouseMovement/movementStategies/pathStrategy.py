import time
import numpy as np
import pyautogui

from ..mouseContext import MouseContext
from .base import MovementStrategy


class FollowPathStrategy(MovementStrategy):
    def __init__(self, path: np.ndarray, interval_s: float = 0.01, steps: int = 10) -> None:
        self.path = path
        self.interval_s = interval_s
        self.steps = steps

    def move(self, context: MouseContext) -> None:
        target = context.target
        screen_size = np.array(pyautogui.size())
        for point1, point2 in zip(self.path[:-1], self.path[1:]):
            point1 = np.clip(np.array(point1) + target, 0, screen_size).astype(int)
            point2 = np.clip(np.array(point2) + target, 0, screen_size).astype(int)
            for i in range(1, self.steps + 1):
                point = point1 + (point2 - point1) * i / self.steps
                pyautogui.moveTo(*point)
                time.sleep(self.interval_s / self.steps)