from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

class MnistClassifierInterface(ABC):

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 1000, max_depth: Optional[int] = None) -> None:
        """
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree (None for unlimited).
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1  # Use all available CPU cores
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Args:
            X_train: Training data, expected shape (n_samples, 784) or (n_samples, 28, 28).
            y_train: Training labels, shape (n_samples,).
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Args:
            X_test: Test data, shape (n_samples, 784) or (n_samples, 28, 28).
        """
        return self.model.predict(X_test)

class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    class MnistModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64
    ) -> None:
        """
        Args:
            learning_rate: Learning rate for the Adam optimizer.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.MnistModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Args:
            X_train: Training data, shape (n_samples, 784) or (n_samples, 28, 28).
            y_train: Training labels, shape (n_samples,).
        """
        # Prepare data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Args:
            X_test: Test data, shape (n_samples, 784) or (n_samples, 28, 28).
        """
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

class ConvolutionalMnistClassifier(MnistClassifierInterface):
    class MnistCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 10)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            return x

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64
    ) -> None:
        """
        Args:
            learning_rate: Learning rate for the Adam optimizer.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.MnistCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Args:
            X_train: Training data, shape (n_samples, 1, 28, 28).
            y_train: Training labels, shape (n_samples,).
        """
        # Prepare data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Args:
            X_test: Test data, shape (n_samples, 1, 28, 28).
        """
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

class MnistClassifier:
    VALID_ALGORITHMS = {'rf', 'nn', 'cnn'}

    def __init__(self, algorithm: str = 'rf') -> None:
        """
        Args:
            algorithm: Algorithm to use ('rf', 'nn', or 'cnn').
        """
        self.algorithm = algorithm.lower()
        if self.algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"Invalid algorithm. Must be one of {self.VALID_ALGORITHMS}")

        if self.algorithm == 'rf':
            self.classifier = RandomForestMnistClassifier()
        elif self.algorithm == 'nn':
            self.classifier = NeuralNetworkMnistClassifier()
        else:  # 'cnn'
            self.classifier = ConvolutionalMnistClassifier()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Args:
            X_train: Training data (shape depends on classifier).
            y_train: Training labels.
        """
        self.classifier.train(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Args:
            X_test: Test data (shape depends on classifier).
        """
        return self.classifier.predict(X_test)

if __name__ == "__main__":
    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Prepare data for Random Forest and Neural Network (flattened)
    X_train_flat = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test_flat = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # Prepare data for CNN (channels-first format)
    X_train_cnn = train_dataset.data.unsqueeze(1).numpy() / 255.0
    X_test_cnn = test_dataset.data.unsqueeze(1).numpy() / 255.0

    # Random Forest
    print("Training Random Forest:")
    rf_classifier = MnistClassifier(algorithm='rf')
    rf_classifier.train(X_train_flat, y_train)
    rf_predictions = rf_classifier.predict(X_test_flat)
    rf_accuracy = np.mean(rf_predictions == y_test)
    print(f"Random Forest Test Acc: {rf_accuracy:.4f}")

    # Neural Network
    print("\nTraining NN:")
    nn_classifier = MnistClassifier(algorithm='nn')
    nn_classifier.train(X_train_flat, y_train)
    nn_predictions = nn_classifier.predict(X_test_flat)
    nn_accuracy = np.mean(nn_predictions == y_test)
    print(f"NN Test Acc: {nn_accuracy:.4f}")

    # Convolutional Neural Network
    print("\nTraining CNN:")
    cnn_classifier = MnistClassifier(algorithm='cnn')
    cnn_classifier.train(X_train_cnn, y_train)
    cnn_predictions = cnn_classifier.predict(X_test_cnn)
    cnn_accuracy = np.mean(cnn_predictions == y_test)
    print(f"CNN TEst Acc: {cnn_accuracy:.4f}")