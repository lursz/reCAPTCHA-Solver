import pyautogui

class GuiAgent:
    def __init__(self) -> None:
        pass
    
    def getScreenResolution(self) -> tuple:
        return pyautogui.size()
    
    def getMousePosition(self) -> tuple:
        return pyautogui.position()
    
    def openBrowser(self, url: str) -> None:
        pyautogui.hotkey('win', 'r')
        pyautogui.typewrite('chrome ' + url)
        pyautogui.press('enter')