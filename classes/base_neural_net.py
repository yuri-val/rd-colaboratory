import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class BaseNeuralNet(nn.Module):
    def __init__(self, tags="base/net", dropout_rate=0.0):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.tags = tags
        self.writer = None
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.init_tensorboard()

    def init_datasets(self, train_dataset, test_dataset):
        self.train_dataset, self.test_dataset = train_dataset, test_dataset

    def init_loaders(self, train_loader, test_loader):
        self.train_loader, self.test_loader = train_loader, test_loader

    def init_tensorboard(self):
        self.writer = SummaryWriter(log_dir=f"runs/{self.class_name}/{self.tags}")

    def close_tensorboard(self):
        if self.writer:
            self.writer.close()

    @staticmethod
    def plot_metrics(train_losses, test_losses, accuracies):
        epochs = range(len(train_losses))
        plt.figure(figsize=(12, 6))

        # Візуалізація втрат
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Візуалізація точності
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies, label="Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.show()

    def build_path(self, version=1):
        tag_path = self.tags.replace("/", "-")
        return f"models/{self.class_name}__{tag_path}__v{version}.pth"

    def save_model(self, version=1):
        save_path = self.build_path(version)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, version=1):
        load_path = self.build_path(version)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Model loaded from {load_path}")

    def show_predictions(self, num_samples=5):
        if not self.test_loader or not self.test_dataset:
            raise ValueError(
                "Test loader and dataset must be initialized before showing predictions."
            )

        self.eval()
        images, labels = next(iter(self.test_loader))
        with torch.no_grad():
            outputs = self(images)
            _, predictions = torch.max(outputs, 1)

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        for i in range(num_samples):
            index = random.randint(0, len(images) - 1)
            img_to_show = (
                images[index].permute(1, 2, 0)
                if images.size(1) == 3
                else images[index].squeeze()
            )
            cmap = None if images.size(1) == 3 else "gray"
            axes[i].imshow(img_to_show, cmap=cmap)
            axes[i].set_title(
                f"True: {self.train_dataset.classes[labels[index]]}\nPred: {self.train_dataset.classes[predictions[index]]}"
            )
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    def log_class_accuracy(self):
        if not self.test_loader or not self.test_dataset:
            raise ValueError(
                "Test loader and dataset must be initialized before logging class accuracy."
            )

        self.eval()
        class_correct = torch.zeros(len(self.test_dataset.classes))
        class_total = torch.zeros(len(self.test_dataset.classes))

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).float()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += correct[i]
                    class_total[label] += 1

        for i, class_name in enumerate(self.test_dataset.classes):
            accuracy = (
                100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            )
            print(f"Accuracy for {class_name}: {accuracy:.2f}%")
            if self.writer:
                self.writer.add_scalar(f"Class Accuracy/{class_name}", accuracy)

    def validate_data(self):
        if not self.train_loader or not self.test_loader:
            raise ValueError("Train and test loaders must be initialized.")
        sample, label = next(iter(self.train_loader))
        print(f"Input shape: {sample.shape}, Label shape: {label.shape}")

    def train_model(
        self,
        epochs=10,
        lr=0.001,
        weight_decay=0.0,
        early_stopping_patience=0,
        criterion=None,
    ):
        """
        Train the neural network model.

        This method performs the training loop for the neural network, including evaluation
        on the test set, early stopping, and logging of metrics.

        Parameters:
        epochs (int): The number of training epochs. Default is 10.
        lr (float): The learning rate for the optimizer. Default is 0.001.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer. Default is 0.0.
        early_stopping_patience (int): The number of epochs to wait for improvement before
                                       stopping training. If 0, early stopping is disabled.
                                       Default is 0.
        criterion (torch.nn.Module): The loss function to use. If None, CrossEntropyLoss is used.
                                     Default is None.

        Raises:
        ValueError: If train and test loaders are not initialized.

        Returns:
        None. The method updates the model in-place and logs the training progress.
        """
        if not self.train_loader or not self.test_loader:
            raise ValueError(
                "Train and test loaders must be initialized before training."
            )

        criterion = criterion or nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses, test_losses, accuracies = [], [], []
        best_loss, patience_counter = float("inf"), 0

        for epoch in range(epochs):
            self.train()
            train_loss = self._train_epoch(criterion, optimizer)
            train_losses.append(train_loss)

            self.eval()
            test_loss, accuracy = self._evaluate_epoch(criterion)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            if early_stopping_patience > 0:
                if test_loss < best_loss:
                    best_loss, patience_counter = test_loss, 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            self._log_to_tensorboard(train_loss, test_loss, accuracy, epoch)

        self.plot_metrics(train_losses, test_losses, accuracies)
        self.show_predictions()
        self.log_class_accuracy()
        self.close_tensorboard()

    def _train_epoch(self, criterion, optimizer):
        running_loss = 0.0
        for images, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = self(images)
            if self.dropout_rate > 0:
                outputs = self.dropout(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def _evaluate_epoch(self, criterion):
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_dataset)
        return test_loss, accuracy

    def _log_to_tensorboard(self, train_loss, test_loss, accuracy, epoch):
        if self.writer:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Test", test_loss, epoch)
            self.writer.add_scalar("Accuracy", accuracy, epoch)
