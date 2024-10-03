import pyautogui
import numpy as np

class MouseEngine:
    def __init__(self) -> None:
        self.target: tuple = None
        self.speed: int = 260.0
        self.max_acceleration: float = 35.5
        self.dt = 1.0
        self.mouse_position = np.array(pyautogui.position(), dtype=float)
        self.mouse_velocity = np.random.rand(2) * 10 - 5
        self.close_distance = 15.0
        
        
    def is_mouse_close(self) -> bool:
        if self.target is None:
            raise ValueError("Mouse Engine: Target is not set")
        
        target_vector = np.array(self.target) - np.array(self.mouse_position)
        target_vector_magnitude = np.linalg.norm(target_vector)
        
        return target_vector_magnitude < self.close_distance
    
    
    def move_mouse_all_the_way(self, target: np.ndarray, speed: float = 30.0) -> None:
        self.target = target
        self.speed = speed
        while not self.is_mouse_close():
            self.move_the_mouse_one_step()            
        pyautogui.click()
        
        
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
        
        self.mouse_velocity += acceleration * self.dt + np.random.normal(0, acceleration_magnitude * 0.3, 2)
        self.mouse_position += self.mouse_velocity * self.dt
        pyautogui.moveTo(*self.mouse_position.astype(int))
        

            
    # def move_mouse_following_path(self, target: np.ndarray, path: np.ndarray, interval_s: float = 0.01) -> None:
    #     N = 10
        
    #     print(path)
        
    #     for point1, point2 in zip(path[:-1], path[1:]):
    #         point1 = np.array(point1) + target
    #         point2 = np.array(point2) + target
            
    #         point1 = np.clip(point1, 0, np.array(pyautogui.size())).astype(int)
    #         point2 = np.clip(point2, 0, np.array(pyautogui.size())).astype(int)
            
    #         print(point1, point2)
            
    #         for i in range(1, N + 1):
    #             point = point1 + (point2 - point1) * i / N
    #             pyautogui.moveTo(*point)
    #             pyautogui.sleep(interval_s / N)
        
        
    