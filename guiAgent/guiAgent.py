import os
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
        
    def closeTab(self) -> None:
        pyautogui.hotkey('ctrl', 'w')
        
    def clickImage(self, image: str) -> None:
        pyautogui.sleep(2)
        pyautogui.click(pyautogui.locateCenterOnScreen(image))
    
    def clickCheckbox(self) -> None:
        self.clickImage('guiAgent/images/captcha_checkbox.png')
        pyautogui.click()
        
    def downloadAudio(self):
        self.clickImage('guiAgent/images/headphones.png')
        self.clickImage('guiAgent/images/download.png')
        
    def takeScreenshot(self) -> None:
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        filename = f'screenshots/screenshot{len(os.listdir("screenshots"))}.png'
        pyautogui.screenshot(filename).crop((50, 50, 1000, 1000)).save(filename)