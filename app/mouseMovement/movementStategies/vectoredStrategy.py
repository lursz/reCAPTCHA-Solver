import numpy as np
import pyautogui

from ..mouseContext import MouseContext
from .base import MovementStrategy


class MoveAllTheWayStrategy(MovementStrategy):
    def move(self, context: MouseContext) -> None:
        while not self._is_mouse_close(context):
            self._move_one_step(context)
        pyautogui.click()

    def _is_mouse_close(self, context: MouseContext) -> bool:
        if context.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        target_vector = np.array(context.target) - context.mouse_position
        return np.linalg.norm(target_vector) < context.close_distance

    def _move_one_step(self, context: MouseContext) -> None:
        if context.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        target_vector = np.array(context.target) - context.mouse_position
        target_magnitude = np.linalg.norm(target_vector)

        MAX_DIST = 400.0
        if target_magnitude > MAX_DIST:
            target_vector = target_vector / target_magnitude * MAX_DIST
        target_velocity = target_vector * context.speed / MAX_DIST
        velocity_diff = target_velocity - context.mouse_velocity
        vel_diff_mag = np.linalg.norm(velocity_diff)
        if vel_diff_mag > context.max_acceleration:
            velocity_diff = velocity_diff / vel_diff_mag * context.max_acceleration
        acceleration = velocity_diff
        noise = np.random.normal(0, np.linalg.norm(acceleration) * 0.3, 2)
        context.mouse_velocity += acceleration * context.dt + noise
        context.mouse_position += context.mouse_velocity * context.dt
        pyautogui.moveTo(*context.mouse_position.astype(int))
