from pyautogui import sleep
from guiAgent.guiAgent import GuiAgent
from mouseEngine.mouseEngine import MouseEngine


def main() -> None:
    print("CAPTCHA Solver")
    # guiAgent = GuiAgent()
    # guiAgent.openBrowser("localhost")
    # guiAgent.clickCheckbox()
    # guiAgent.takeScreenshot()
    # guiAgent.closeTab()
    mouse = MouseEngine()
    mouse.target = (300, 300)
    while True:
        mouse.move_the_mouse_one_step()
        # sleep(0.001)
    

if __name__ == '__main__':
    main()