import numpy as np
from dataclasses import dataclass, field
import pyautogui


@dataclass
class MouseContext:
    target: np.ndarray = None
    speed: float = 80.0
    max_acceleration: float = 35.5
    dt: float = 1.0
    mouse_position: np.ndarray = field(default_factory=lambda: np.array(pyautogui.position(), dtype=float))
    mouse_velocity: np.ndarray = field(default_factory=lambda: np.random.rand(2) * 10 - 5)
    close_distance: float = 15.0


