from torch.utils.data import DataLoader, Dataset
import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class MouseMovementDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = [torch.tensor(d, dtype=torch.float32).to(device) for d in data]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]
