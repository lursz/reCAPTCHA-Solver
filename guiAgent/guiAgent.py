import os
import pyautogui

class GuiAgent:
    def __init__(self) -> None:
        pass
    
    def get_screen_resolution(self) -> tuple:
        return pyautogui.size()
    
    def get_mouse_position(self) -> tuple:
        return pyautogui.position()
    
    def open_browser(self, url: str) -> None:
        pyautogui.hotkey('win', 'r')
        pyautogui.typewrite('brave --incognito' + url)
        pyautogui.press('enter')
        print("Browser opened")
        
    def close_tab(self) -> None:
        pyautogui.hotkey('ctrl', 'w')
        
    def click_image(self, image: str) -> None:
        pyautogui.sleep(2)
        pyautogui.click(pyautogui.locateCenterOnScreen(image))
    
    def click_checkbox(self) -> None:
        self.click_image('guiAgent/images/captcha_checkbox.png')
        pyautogui.click()
        
    def download_audio(self) -> None:
        self.click_image('guiAgent/images/headphones.png')
        self.click_image('guiAgent/images/download.png')
        
    def take_screenshot(self) -> str:
        if not os.path.exists('guiAgent/screenshots'):
            os.makedirs('guiAgent/screenshots')
        filename = f'guiAgent/screenshots/screenshot{len(os.listdir("screenshots"))}.png'
        pyautogui.screenshot(filename).crop((50, 50, 1000, 1000)).save(filename)
        return filename