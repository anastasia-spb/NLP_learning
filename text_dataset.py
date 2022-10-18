from torch.utils.data import Dataset
import pandas as pd


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index >= len(self.df):
            raise IndexError
        text = self.df.text[index]
        label = -1
        if 'label' in self.df:
            label = self.df.label[index]
        if self.transform is not None:
            text = self.transform(text)
        return {"text": text, "label": label}
