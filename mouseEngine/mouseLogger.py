import tkinter as tk
import json
import time
import random
import pyautogui
from threading import Timer

class MouseLoggerApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self.export_to_json_and_quit)
        
        self.canvas = tk.Canvas(root, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        
        self.canvas.update()  # Ensure canvas is updated to get correct width and height
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        
        self.green_square = self.create_random_square('green')
        self.red_square = self.create_random_square('red')
        
        self.logging = False
        self.mouse_data = []
        self.current_mouse_data = []
        
        self.canvas.tag_bind(self.green_square, "<Button-1>", self.start_logging)
        self.canvas.tag_bind(self.red_square, "<Button-1>", self.stop_logging)
        
    def create_random_square(self, color):
        size = 30
        print(self.canvas_width, self.canvas_height)
        x1 = random.randint(0, self.canvas_width - size)
        y1 = random.randint(0, self.canvas_height - size)
        x2 = x1 + size
        y2 = y1 + size
        
        self.rect_x = (x1 + x2) // 2
        self.rect_y = (y1 + y2) // 2
        
        return self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        
    def start_logging(self, event):
        if self.logging:
            return
        
        self.logging = True
        self.log_mouse_position(event, len(self.mouse_data))
        
    def stop_logging(self, event):
        self.logging = False
        self.mouse_data.append(self.current_mouse_data)
        self.current_mouse_data = []
        
        self.canvas.delete(self.green_square)
        self.canvas.delete(self.red_square)
        
        self.green_square = self.create_random_square('green')
        self.red_square = self.create_random_square('red')
        
        self.canvas.tag_bind(self.green_square, "<Button-1>", self.start_logging)
        self.canvas.tag_bind(self.red_square, "<Button-1>", self.stop_logging)
        
        
    def log_mouse_position(self, event, index):
        if self.logging:       
            # print("Logging ", time.time())
            has_history = len(self.current_mouse_data) > 0
            
            current_x = (pyautogui.position().x - self.rect_x) / self.canvas_width
            current_y = (pyautogui.position().y - self.rect_y) / self.canvas_height
            
            last_time_stamp = self.current_mouse_data[-1]['timestamp'] if has_history else time.time()
            last_x = self.current_mouse_data[-1]['x'] if has_history else current_x
            last_y = self.current_mouse_data[-1]['y'] if has_history else current_y
            
            if not has_history or last_x != current_x or last_y != current_y:
                self.current_mouse_data.append({
                    'timestamp': time.time(),
                    'speed': ((current_x - last_x) ** 2 + (current_y - last_y) ** 2) ** 0.5 / (time.time() - last_time_stamp) if has_history else 0,
                    'x': current_x,
                    'y': current_y
                })
            
            if index == len(self.mouse_data):
                Timer(0.01, lambda: self.log_mouse_position(event, index)).start()
        
    def export_to_json_and_quit(self, events):
        with open('mouse_data.json', 'w') as file:
            json.dump(self.mouse_data, file, indent=4)
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = MouseLoggerApp(root)
    root.mainloop()
