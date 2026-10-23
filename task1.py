import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class CsvDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)
        self.features = torch.tensor(
            df.iloc[:, :-1].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            df.iloc[:, -1].values, dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def main():
    dataset = CsvDataset("cancer.csv")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset, [train_size, val_size]
    )

    for workers in [0, 2]:
        loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            num_workers=workers
        )

        batches = 0
        for features, labels in loader:
            if batches == 0:
                print("Batch shape:", features.shape)
            batches += 1

        print("Workers:", workers, "| Batches:", batches)


if __name__ == "__main__":
    main()