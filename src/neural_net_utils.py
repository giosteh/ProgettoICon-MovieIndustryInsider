import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from supervised_utils import prepare_data

sns.set_style("darkgrid")

# setting del device
device = "cuda" if torch.cuda.is_available() else "cpu"



def build_mlp_architecture(input_dim):
    """
    Definisce l'architettura della rete.
    """
    layer_dims = [input_dim, 32, 16, 8]

    layers_list = []
    for i in range(len(layer_dims) - 1):
        layers_list.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        layers_list.append(nn.ReLU())
        if i > 0:
            layers_list.append(nn.Dropout(p=.2))

    return layers_list


class RegressionMLP(nn.Module):
    """
    Rete per il task di regressione.
    """

    def __init__(self, input_dim):
        super(RegressionMLP, self).__init__()

        layers_list = build_mlp_architecture(input_dim)
        layers_list.append(nn.Linear(8, 1))
        self.mlp = nn.Sequential(*layers_list)
        # inizializzazione dei pesi
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inizializza i pesi della rete.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward propagation.
        """
        y = self.mlp(x)
        return y


class ClassificationMLP(nn.Module):
    """
    Rete per il task di classificazione.
    """

    def __init__(self, input_dim, num_classes):
        super(ClassificationMLP, self).__init__()

        layers_list = build_mlp_architecture(input_dim)
        layers_list.append(nn.Linear(8, num_classes))
        self.mlp = nn.Sequential(*layers_list)
        # inizializzazione dei pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inizializza i pesi della rete.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward propagation.
        """
        y = self.mlp(x)
        return y


class EarlyStopping:
    """
    Early stopping per l'addestramento della rete.
    """

    def __init__(self, model, model_name, patience=30, dir_path="models", mode="min"):
        self._patience = patience
        self._model = model
        self._counter = 0
        self._best_score = None
        self._stop = False
        self.mode = mode
        self._checkpoint_path = f"{dir_path}/{model_name}"

        self._train_scores = []
        self._val_scores = []
    
    def _is_improvement(self, score):
        """
        Controlla se lo score migliora.
        """
        if self.mode == "max":
            return score > self._best_score
        else:
            return score < self._best_score
    
    def __call__(self, train_score, val_score):
        """
        Verifica se il modello migliora o meno.
        """
        self._train_scores.append(train_score)
        self._val_scores.append(val_score)
        score = val_score

        if self._best_score is None:
            self._best_score = score
        elif self._is_improvement(score):
            self._save_checkpoint()
            self._best_score = score
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stop = True
                self.plot()
        
        return self._stop

    def _save_checkpoint(self):
        """
        Salva il modello.
        """
        torch.save(self._model.state_dict(), self._checkpoint_path)
    
    def plot(self):
        """
        Visualizza in un grafico la progressione del training.
        """
        score_name = "Accuracy" if self.mode == "max" else "Loss"
        plt.figure(figsize=(7, 5))
        plt.plot(self._train_scores, label=f"Train {score_name}", color="dodgerblue", linestyle="dashed", linewidth=2.3)
        plt.plot(self._val_scores, label=f"Val {score_name}", color="crimson", linewidth=2.3)
        plt.title("Model Performance")
        plt.xlabel("Epoch")
        plt.ylabel(score_name)

        plt.legend()
        plt.show()
        plt.close()

class Trainer:
    """
    Trainer per l'addestramento della rete.
    """

    def __init__(self, df, cols, features_subset=None, resample=False, task="regression", num_classes=3):
        self._task = task
        self._train_loader, self._val_loader, self._test_loader, self._input_dim = self._get_data_loaders(df, cols, features_subset, resample, task)
        self._model = RegressionMLP(self._input_dim) if task == "regression" else ClassificationMLP(self._input_dim, num_classes)
        self._model.to(device)

        # setup per il training
        self._criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-2, weight_decay=1e-2)

        self._model_name = f"{task.capitalize()}_MLP.pt"
        self._early_stopping = EarlyStopping(self._model, self._model_name)
        self._early_stopping.mode = "min" if task == "regression" else "max"

    def _get_data_loaders(self, df, cols, features_subset=None, resample=False,
                          task="regression", val_split=.2, batch_size=32):
        """
        Prepara i data loaders per l'addestramento della rete.
        """
        X_train, X_test, y_train, y_test = prepare_data(df, cols, resample=resample, task=task)

        if features_subset:
            X_train = X_train[features_subset]
            X_test = X_test[features_subset]

        if task == "regression":
            train_dataset = TensorDataset(
                torch.tensor(X_train.values, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
            test_dataset = TensorDataset(
                torch.tensor(X_test.values, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1))
        else:
            train_dataset = TensorDataset(
                torch.tensor(X_train.values, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long))
            test_dataset = TensorDataset(
                torch.tensor(X_test.values, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long))
        
        train_size = int(len(train_dataset) * (1 - val_split))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader, X_train.shape[1]
    
    def _train(self):
        """
        Addestra il modello per una singola epoca.
        """
        self._model.train()
        correct = 0
        size = len(self._train_loader.dataset)

        total_loss = .0
        for features, labels in self._train_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = self._model(features)
            loss = self._criterion(logits, labels)

            # backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

            if self._task == "classification":
                predicted = logits.argmax(dim=-1)
                correct += (predicted == labels).sum().item()

        total_loss /= size
        if self._task == "classification":
            accuracy = 100 * correct / size
            return total_loss, accuracy
        else:
            return total_loss, None

    def _validate(self):
        """
        Valida il modello.
        """
        self._model.eval()
        correct = 0
        size = len(self._val_loader.dataset)

        total_loss = .0
        with torch.no_grad():
            for features, labels in self._val_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits = self._model(features)
                loss = self._criterion(logits, labels)

                total_loss += loss.item()

                if self._task == "classification":
                    predicted = logits.argmax(dim=-1)
                    correct += (predicted == labels).sum().item()

        total_loss /= size
        if self._task == "classification":
            accuracy = 100 * correct / size
            return total_loss, accuracy
        else:
            return total_loss, None

    def test(self):
        """
        Testa il modello.
        """
        self._model.eval()
        correct = 0
        size = len(self._test_loader.dataset)

        total_loss = .0
        with torch.no_grad():
            for features, labels in self._test_loader:
                features = features.to(device)
                labels = labels.to(device)

                logits = self._model(features)
                loss = self._criterion(logits, labels)

                total_loss += loss.item()

                if self._task == "classification":
                    predicted = logits.argmax(dim=-1)
                    correct += (predicted == labels).sum().item()
        
        loss = total_loss / size
        if self._task == "classification":
            accuracy = 100 * correct / size
            return loss, accuracy
        return loss, None

    def fit(self, epochs=100, verbose=True):
        """
        Addestra il modello.
        """
        if verbose:
            print("> TRAINING...\n")
        for epoch in range(epochs):
            train_loss, train_acc = self._train()
            val_loss, val_acc = self._validate()

            if self._task == "classification":
                train_score, val_score = train_acc, val_acc
            else:
                train_score, val_score = train_loss, val_loss
            
            if verbose:
                print(f"Epoch #{epoch+1}")
                if self._task == "classification":
                    print(f"Train accuracy: {train_acc:.2f}% | Val accuracy: {val_acc:.2f}%\n")
                else:
                    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")
            # check per l'early stopping
            stop = self._early_stopping(train_score, val_score)
            if stop:
                if verbose:
                    print(f"Early stopping at epoch #{epoch+1}.\n")
                break
        # testing del modello
        if verbose:
            print("> TESTING...\n")
        test_loss, test_acc = self.test()
        if self._task == "classification":
            print(f"Accuracy: {test_acc:.2f}%")
        else:
            print(f"Loss: {test_loss:.4f}\n")
