import pyautogui
import numpy as np

class MouseEngine:
    def __init__(self) -> None:
        self.target: tuple = None
        self.speed: int = 30.0
        self.max_acceleration: float = 1.5
        self.dt = 1.0
        self.mouse_position = np.array(pyautogui.position(), dtype=float)
        self.mouse_velocity = np.random.rand(2) * 10 - 5
        
    def move_the_mouse_one_step(self) -> None:
        if self.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        
        target_vector = np.array(self.target) - np.array(self.mouse_position)
        target_vector_magnitude = np.linalg.norm(target_vector) #vector length
        
        MAX_DIST = 400.0
        
        # if close then scale vector to max distance
        if target_vector_magnitude > MAX_DIST:
            target_vector = target_vector / np.linalg.norm(target_vector) * MAX_DIST
            
        target_velocity = target_vector * self.speed / MAX_DIST # eventual speed in infinite time
        print(target_velocity, self.mouse_velocity)
        
        velocity_difference = target_velocity - self.mouse_velocity
        velocity_difference_magnitude = np.linalg.norm(velocity_difference)
        
        # if the difference is too big, scale it to max acceleration
        if velocity_difference_magnitude > self.max_acceleration:
            velocity_difference = velocity_difference / velocity_difference_magnitude * self.max_acceleration
            
        acceleration = velocity_difference
        acceleration_magnitude = np.linalg.norm(acceleration)
        
        self.mouse_velocity += acceleration * self.dt + np.random.normal(0, acceleration_magnitude, 2)
        self.mouse_position += self.mouse_velocity * self.dt
        pyautogui.moveTo(*self.mouse_position.astype(int))
        
        
        
        
        
        
    