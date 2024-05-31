from guiAgent.guiAgent import GuiAgent


def main() -> None:
    print("CAPTCHA Solver")
    guiAgent = GuiAgent()
    guiAgent.openBrowser("localhost")
    guiAgent.clickCheckbox()
    # guiAgent.downloadAudio()

    


if __name__ == '__main__':
    main()