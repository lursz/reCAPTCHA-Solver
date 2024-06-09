import tkinter as tk
import json
import time
import random
import pyautogui

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
        return self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
        
    def start_logging(self, event):
        self.logging = True
        self.log_mouse_position(event)
        
    def stop_logging(self, event):
        self.logging = False
        self.mouse_data.append(self.current_mouse_data)
        
        self.canvas.delete(self.green_square)
        self.canvas.delete(self.red_square)
        
        self.green_square = self.create_random_square('green')
        self.red_square = self.create_random_square('red')
        
        self.canvas.tag_bind(self.green_square, "<Button-1>", self.start_logging)
        self.canvas.tag_bind(self.red_square, "<Button-1>", self.stop_logging)
        
        
    def log_mouse_position(self, event):
        if self.logging:
            self.current_mouse_data.append({
                'timestamp': time.time(),
                'x': pyautogui.position().x,
                'y': pyautogui.position().y
            })
            self.root.after(10, lambda: self.log_mouse_position(event))
        
    def export_to_json_and_quit(self, events):
        with open('mouse_data.json', 'w') as file:
            json.dump(self.mouse_data, file, indent=4)
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = MouseLoggerApp(root)
    root.mainloop()
