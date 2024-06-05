import torchsummary

class CustomModel:
    def printModelSummary(self):
        print(torchsummary.summary(self, (3, 150, 150)))