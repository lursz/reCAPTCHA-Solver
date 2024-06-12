from pyautogui import sleep
from guiAgent.guiAgent import GuiAgent
from imageProcessing.imageProcessor import ImageProcessor
from mouseEngine.mouseEngine import MouseEngine


def main() -> None:
    print("CAPTCHA Solver")
    gui_agent = GuiAgent()
    gui_agent.openBrowser("localhost")
    gui_agent.clickCheckbox()
    gui_agent.takeScreenshot()
    gui_agent.closeTab()
    
    image_processor = ImageProcessor("screenshot.png")
    image_processor.show_image()
    list_of_img =  image_processor.process_image()
    
    
    mouse = MouseEngine()
    mouse.target = (300, 300)
    while True:
        mouse.move_the_mouse_one_step()
        # sleep(0.001)
    

if __name__ == '__main__':
    main()