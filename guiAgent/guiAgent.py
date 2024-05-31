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
        pyautogui.typewrite('brave ' + url)
        pyautogui.press('enter')
        print("Browser opened")
        
    def clickImage(self, image: str) -> None:
        pyautogui.click(pyautogui.locateCenterOnScreen(image))
    
    def clickCheckbox(self) -> None:
        self.clickImage('images/captcha_checkbox.png')
        
    def downloadAudio(self):
        self.clickImage('images/headphones.png')
        self.clickImage('images/download.png')
