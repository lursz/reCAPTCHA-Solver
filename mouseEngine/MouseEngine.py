import pyautogui
import numpy as np

class MouseEngine:
    def __init__(self) -> None:
        self.target: tuple = None
        self.speed: int = 10
        self.max_acceleration: float = 2.0
        self.dt = 1.0
        self.mouse_position = np.array(pyautogui.position())
        self.mouse_velocity = np.array([0, 0])
        
    def setPathForMouse(self) -> None:
        if self.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        
        target_vector = np.array(self.target) - np.array(self.mouse_position)
        target_vector = target_vector / np.linalg.norm(target_vector)
        target_velocity = target_vector * self.speed
        
        velocity_difference = target_velocity - self.mouse_velocity
        velocity_difference_magnitude = np.linalg.norm(velocity_difference)
        
        if velocity_difference_magnitude > self.max_acceleration:
            velocity_difference = velocity_difference / velocity_difference_magnitude * self.max_acceleration
            
        acceleration = velocity_difference
        
        self.mouse_velocity += acceleration * self.dt
        self.mouse_position += self.mouse_velocity * self.dt
        
        
        
    