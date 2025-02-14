import pickle
from matplotlib import pyplot as plt
import numpy as np
from torchinfo import summary
class ModelTools:
    def print_model_summary(self):
        print(summary(self, (1, 3, 150, 150)))
        
    def plot_accuracy_from_history(self, *histories, labels=None, path=None) -> None:
        plt.rcParams['figure.figsize'] = (25.0, 5.0)  # set default size of plots

        for i, history in enumerate(histories):
            epochs = np.arange(1, len(history['accuracy']) + 1)
            
            color = ['b', 'r', 'g', 'c', 'm', 'y'][i % 6]  # Choose color cyclically
            label = labels[i] if labels else f'Model {i+1}'
            acc = [acc.cpu().item() for acc in history['accuracy']]
            val_acc = [val_acc.cpu().item() for val_acc in history['val_accuracy']]
            
            plt.plot(epochs, acc, color + 'o', label=f'Training accuracy for {label}')
            plt.plot(epochs, val_acc, color, label=f'Validation accuracy for {label}')

        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if path:
            plt.savefig(path)
        else:
            plt.show()
            
    def save_learning_data_to_pickle(self, history: dict, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(history, file)