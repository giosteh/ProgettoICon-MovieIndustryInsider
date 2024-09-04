import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from supervised_utils import prepare_data

sns.set_style('whitegrid')

# setting del device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# funzione che definisce l'architettura della rete
def define_net_architecture(input_dim):
    # definisco l'architettura della rete
    layers = []
    if input_dim <= 60:
        layers = [input_dim, 64, 32, 16, 8]
    else:
        layers = [input_dim, 128, 64, 32, 16, 8]
    
    # costruisco la rete
    layers_list = []
    for i in range(len(layers) - 1):
        layers_list.append(nn.Linear(layers[i], layers[i + 1]))
        layers_list.append(nn.ReLU())
        if i > 0:
            layers_list.append(nn.Dropout(p=0.3))

    return layers_list


# neural net per il task di regressione
class RegressionNet(nn.Module):

    def __init__(self, input_dim):
        super(RegressionNet, self).__init__()

        # ottengo l'architettura della rete
        layers_list = define_net_architecture(input_dim)
        layers_list.append(nn.Linear(8, 1))

        self.net = nn.Sequential(*layers_list)

        # inizializzazione dei pesi
        self._initialize_weights()

    # metodo per l'inizializzazione dei pesi
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    # metodo di forward propagation
    def forward(self, x):
        y = self.net(x)
        return y


# neural net per il task di classificazione
class ClassificationNet(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(ClassificationNet, self).__init__()

        # ottengo l'architettura della rete
        layers_list = define_net_architecture(input_dim)
        layers_list.append(nn.Linear(8, num_classes))

        self.net = nn.Sequential(*layers_list)

        # inizializzazione dei pesi
        self._initialize_weights()
    
    # metodo per l'inizializzazione dei pesi
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    # metodo di forward propagation
    def forward(self, x):
        y = self.net(x)
        return y


# classe che implementa l'early-stopping
class EarlyStopping:

    def __init__(self, patience=5, delta=0, verbose=True, path='net/model.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.val_loss_min = float('inf')
        self.best_score = None
        self.early_stop = False

        self.val_scores = []
        self.train_scores = []
    
    # funzione che implementa l'early-stopping
    def __call__(self, model, val_loss, train_loss):
        self.val_scores.append(val_loss)
        self.train_scores.append(train_loss)

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early-stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    # funzione che salva il modello
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Val loss ({self.val_loss_min:.4f} --> {val_loss:.4f})')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    # funzione che plotta i risultati
    def plot_scores(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_scores, label='Train loss', linestyle='dashed', linewidth=2)
        plt.plot(self.val_scores, label='Validation loss', linewidth=2.5)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Loss per epoch')
        plt.legend()

        plt.show()


# funzione per il training della rete
def train_net(model, train_loader, loss_fn, optimizer):
    size = len(train_loader)
    running_loss = .0

    model.train()
    for X, y in train_loader:
        # forward
        pred = model(X)
        loss = loss_fn(pred, y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # ritorno la training loss media
    return running_loss / size


# funzione per la validation della rete
def validate_net(model, val_loader, loss_fn):
    size = len(val_loader)
    val_loss = .0

    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            # forward
            pred = model(X)
            loss = loss_fn(pred, y)

            val_loss += loss.item()
    
    # ritorno la validation loss media
    return val_loss / size


# funzione per il test della rete
def test_net(model, test_loader, loss_fn, task='regression'):
    size = len(test_loader)
    total = len(test_loader.dataset)
    test_loss = .0
    correct = 0

    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            # forward
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()

            if task == 'classification':
                _, pred = torch.max(pred, dim=1)
                correct += (pred == y).sum().item()

                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

    # calcolo della test loss media
    test_loss = test_loss / size

    if task == 'classification':
        # stampo il classification report
        print_report(labels, preds)

        accuracy = 100 * correct / total
        return test_loss, accuracy

    return test_loss


# funzione che stampa il classification report
def print_report(labels, preds):
    report = classification_report(labels, preds, output_dict=True,
                                   target_names=['Classe 0', 'Classe 1', 'Classe 2'])
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])

    plt.figure(figsize=(8, 5))
    sns.heatmap(report_df.iloc[:-3, :], annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.show()
    print('\n')


# funzione che ottiene i data loader per training, validation e test
def get_data_loaders(df, cols, batch_size=32, val_split=0.2, resample=False, task='regression'):
    # ottengo X e y
    X_train, X_test, y_train, y_test = prepare_data(df, cols['target'], cols['drop'], cols['dummies'], cols['labels'],
                                                    cols['round'], cols['clipping'], cols['standardize'], cols['minmax'],
                                                    resample=resample, task=task)
    input_dim = len(X_train.columns)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_size = int(len(train_dataset) * (1 - val_split))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # ottengo i data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, input_dim


# funzione per il tuning e test della rete
def train_and_test_net(df, cols, epochs=100, val_split=0.2, resample=False, task='regression'):
    # data loaders per training, validation e test
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(df, cols, val_split=val_split, resample=resample, task=task)

    loss_fn = None
    model = None

    # inizializzo la rete
    if task == 'regression':
        model = RegressionNet(input_dim)
        loss_fn = nn.MSELoss()
    elif task == 'classification':
        model = ClassificationNet(input_dim, 3)
        loss_fn = nn.CrossEntropyLoss()
    
    model = model.to(device)

    # inizializzo l'optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # inizializzo l'early stopping
    path = f'net/model-{task}.pt'
    early_stopping = EarlyStopping(patience=5, verbose=True, path=path)

    # training della rete
    for epoch in range(epochs):
        train_loss = train_net(model, train_loader, loss_fn, optimizer)
        val_loss = validate_net(model, val_loader, loss_fn)

        print(f'Epoch {epoch + 1} | train loss: {train_loss}, val loss: {val_loss}')

        # early stopping
        early_stopping(model, val_loss, train_loss)
        if early_stopping.early_stop:
            print(f'\nEarly stopping at epoch {epoch}.')
            break
    
    # plotto l'evoluzione della loss durante il training
    early_stopping.plot_scores()

    # carico il modello migliore
    model.load_state_dict(torch.load(path))

    # test della rete
    if task == 'regression':
        test_loss = test_net(model, test_loader, loss_fn)
        print(f'\nTest loss: {test_loss:.4f}')
    elif task == 'classification':
        test_loss, accuracy = test_net(model, test_loader, loss_fn, task=task)
        print(f'\nTest accuracy: {accuracy:.2f}%\nTest loss: {test_loss:.4f}')
