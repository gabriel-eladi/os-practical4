import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Logger:
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

    def log(self, epoch, train_loss, val_loss, val_acc):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

    def plot(self):
        pass

    def flush(self):
        pass

class beanNeuralNetwork(nn.Module):
    def __init__(self, trainingDataset: pd.DataFrame, testingDataset: pd.DataFrame, 
                 input_size=50, hidden_size=100, output_size=16, 
                 learning_rate=0.2, batch_size=50, validation_share=0.2):
        super(beanNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.function1 = nn.Linear(input_size, hidden_size)
        self.batchNorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.function2 = nn.Linear(hidden_size, output_size)
        self.softMax = nn.Softmax(dim=1)

        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_share = validation_share

        # Convert dfs to tensors
        self.label_encoder = LabelEncoder()


        scaler = StandardScaler()

        self.trainingTensor = trainingDataset.drop(columns="Class").to_numpy()
        self.trainingTensor = scaler.fit_transform(self.trainingTensor)
        self.trainingTensor = torch.from_numpy(self.trainingTensor).float()

        training_labels = self.label_encoder.fit_transform(trainingDataset["Class"])
        self.trainingLabels = torch.from_numpy(training_labels).long()

        self.testingTensor = testingDataset.to_numpy()
        self.testingTensor = scaler.transform(self.testingTensor)
        self.testingTensor = torch.from_numpy(self.testingTensor).float()

        self.train_number = len(trainingDataset)
        self.train_indices = list(range(self.train_number))
        # Every day I'm shufflin
        np.random.shuffle(self.train_indices)
        self.testing_portion = round(self.train_number * validation_share) #probably 2000 or so
        self.training_index = self.train_indices[self.testing_portion:]
        self.testing_index = self.train_indices[:self.testing_portion]

    def forward(self, x):
        x = self.function1(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        x = self.function2(x)
        # CrossEntropyLoss has softmax already, uncomment this when we plug in our own loss function
        # x = self.softMax(x)
        return x
    
    def train_model(self, epochs=50):
        # Get features X and labels y
        beans = self.trainingTensor[:, :]
        labels = self.trainingLabels

        beans = torch.as_tensor(beans, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        # Create training/validation datasets
        train_dataset = torch.utils.data.TensorDataset(
            beans[self.training_index],
            labels[self.training_index]
        )
        val_dataset = torch.utils.data.TensorDataset(
            beans[self.testing_index],
            labels[self.testing_index]
        )

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        #==================================================================================================
        # Loss function (plug new one here)
        criterion = nn.CrossEntropyLoss()
        #==================================================================================================
        
        # Optimizer (the thing changing the neuron weights)
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0.0

            for bean_batch, label_batch in train_loader:
                # Remove old gradients
                optimizer.zero_grad()
                # Forward pass
                predictions = self(bean_batch)
                # Calculate loss
                loss = criterion(predictions, label_batch)
                # Backpropagation
                loss.backward()
                # Update weights
                optimizer.step()

                total_train_loss += loss.item()

            # Average training loss
            train_loss = total_train_loss / len(train_loader)

            # Validation loop
            self.eval()
            total_val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for bean_batch, label_batch in valid_loader:
                    # Make predictions for the batch
                    predictions = self(bean_batch)
                    loss = criterion(predictions, label_batch)
                    total_val_loss += loss.item()
                    # Get predicted class (the one with the highest confidence)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    correct += (predicted_classes == label_batch).sum().item()
                    total += label_batch.size(0)

            valid_loss = total_val_loss / len(valid_loader)
            valid_acc = correct / total

            print(f"Epoch {epoch + 1}/{epochs} | "f"Training Loss: {train_loss:.4f} | "
                  f"Validation Loss: {valid_loss:.4f} | "f"Validation Accuracy: {valid_acc:.4f}")


#==================================================================================================
training_df = pd.read_csv("dry_bean_train.csv")
test_df = pd.read_csv("dry_bean_test.csv")

mlp = beanNeuralNetwork(
    trainingDataset=training_df,
    testingDataset=test_df,
    input_size=16,
    hidden_size=100,
    output_size=16,
    learning_rate=0.01,
    batch_size=50,
    validation_share=0.2
)
mlp.train_model(epochs=50)