import pyautogui
import numpy as np
from abc import ABC, abstractmethod
import time

class MovementStrategy(ABC):
    @abstractmethod
    def move(self, mouse_engine):
        pass

class FollowPathStrategy(MovementStrategy):
    def __init__(self, path, interval_s=0.01, steps=10):
        self.path = path
        self.interval_s = interval_s
        self.steps = steps

    def move(self, mouse_engine):
        target = mouse_engine.target
        for point1, point2 in zip(self.path[:-1], self.path[1:]):
            point1 = np.clip(np.array(point1) + target, 0, np.array(pyautogui.size())).astype(int)
            point2 = np.clip(np.array(point2) + target, 0, np.array(pyautogui.size())).astype(int)
            for i in range(1, self.steps + 1):
                point = point1 + (point2 - point1) * i / self.steps
                pyautogui.moveTo(*point)
                time.sleep(self.interval_s / self.steps)
                
class MoveAllTheWayStrategy(MovementStrategy):
    def move(self, mouse_engine):
        while not self._is_mouse_close(mouse_engine):
            self._move_one_step(mouse_engine)
        pyautogui.click()
    
    def _is_mouse_close(self, mouse_engine) -> bool:
        if mouse_engine.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        target_vector = np.array(mouse_engine.target) - mouse_engine.mouse_position
        return np.linalg.norm(target_vector) < mouse_engine.close_distance

    def _move_one_step(self, mouse_engine) -> None:
        if mouse_engine.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        target_vector = np.array(mouse_engine.target) - mouse_engine.mouse_position
        target_magnitude = np.linalg.norm(target_vector)
        
        MAX_DIST = 400.0
        if target_magnitude > MAX_DIST:
            target_vector = target_vector / target_magnitude * MAX_DIST
        target_velocity = target_vector * mouse_engine.speed / MAX_DIST
        velocity_diff = target_velocity - mouse_engine.mouse_velocity
        vel_diff_mag = np.linalg.norm(velocity_diff)
        if vel_diff_mag > mouse_engine.max_acceleration:
            velocity_diff = velocity_diff / vel_diff_mag * mouse_engine.max_acceleration
        acceleration = velocity_diff
        mouse_engine.mouse_velocity += acceleration * mouse_engine.dt + np.random.normal(0, np.linalg.norm(acceleration) * 0.3, 2)
        mouse_engine.mouse_position += mouse_engine.mouse_velocity * mouse_engine.dt
        pyautogui.moveTo(*mouse_engine.mouse_position.astype(int))


class MouseEngine:
    def __init__(self) -> None:
        self.target = None
        self.speed: float = 80.0
        self.max_acceleration: float = 35.5
        self.dt = 1.0
        self.mouse_position = np.array(pyautogui.position(), dtype=float)
        self.mouse_velocity = np.random.rand(2) * 10 - 5
        self.close_distance = 15.0
        self.strategy: MovementStrategy = None

    def set_strategy(self, strategy: MovementStrategy):
        self.strategy = strategy

    def move_mouse(self):
        if self.strategy:
            self.strategy.move(self)
        else:
            raise ValueError("Movement strategy not set")

    def move_mouse_all_the_way(self, position) -> None:
        self.target = position
        self.set_strategy(MoveAllTheWayStrategy())
        self.move_mouse()
