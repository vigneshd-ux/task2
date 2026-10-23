
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


# 1. Create CSV dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["label"] = data.target
df.to_csv("cancer.csv", index=False)


# 2. Dataset class
class CsvDataset(Dataset):
    def __init__(self, file):
        df = pd.read_csv(file)
        self.x = torch.tensor(
            df.iloc[:, :-1].values, dtype=torch.float32
        )
        self.y = torch.tensor(
            df.iloc[:, -1].values, dtype=torch.long
        )

        # Scale features for better training
        scaler = StandardScaler()
        self.x = torch.tensor(
            scaler.fit_transform(self.x.numpy()),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 3. MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


def main():
    torch.manual_seed(42)

    dataset = CsvDataset("cancer.csv")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=32, shuffle=False, num_workers=0
    )

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Train for 30 epochs
    for epoch in range(30):
        model.train()

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # 5. Evaluate model
    def accuracy(model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                outputs = model(features)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return 100 * correct / total

    before = accuracy(model, val_loader)
    print(f"Before saving: {before:.2f}%")

    # 6. Save model weights
    torch.save(model.state_dict(), "model.pt")

    # 7. Create new model and load weights
    new_model = MLP()
    new_model.load_state_dict(
        torch.load("model.pt", map_location="cpu", weights_only=True)
    )
    new_model.eval()

    after = accuracy(new_model, val_loader)
    print(f"After loading:  {after:.2f}%")
    print("Identical accuracy:", before == after)

    # 8. Predict one new row
    def predict(features):
        new_model.eval()
        features = torch.tensor(
            features, dtype=torch.float32
        ).reshape(1, -1)

        with torch.no_grad():
            probabilities = torch.softmax(
                new_model(features), dim=1
            )
            probability, label = probabilities.max(dim=1)

        return label.item(), probability.item()

    sample, _ = dataset[0]
    label, probability = predict(sample.tolist())
    print("Predicted label:", label)
    print(f"Probability: {probability:.4f}")

    # 9. Bonus: Save full training checkpoint
    torch.save({
        "epoch": 30,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "checkpoint.pt")

    # Example: resume training from checkpoint
    checkpoint = torch.load(
        "checkpoint.pt", map_location="cpu", weights_only=True
    )
    resume_model = MLP()
    resume_model.load_state_dict(checkpoint["model_state_dict"])

    resume_optimizer = torch.optim.Adam(
        resume_model.parameters(), lr=0.001
    )
    resume_optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]
    print("Resume from epoch:", start_epoch)


if __name__ == "__main__":
    main()
