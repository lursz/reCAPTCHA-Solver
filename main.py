from pyautogui import sleep
from guiAgent.guiAgent import GuiAgent


def main() -> None:
    print("CAPTCHA Solver")
    guiAgent = GuiAgent()
    guiAgent.openBrowser("localhost")
    guiAgent.clickCheckbox()
    sleep(5)
    guiAgent.takeScreenshot()
    guiAgent.closeTab()

    

if __name__ == '__main__':
    main()