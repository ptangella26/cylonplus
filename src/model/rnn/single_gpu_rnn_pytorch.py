import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib.pyplot as plt

class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def load_and_process_data(filename, sequence_length):
    temp = pd.read_csv("Data/temperature.csv")
    temp_NY = temp[['datetime', 'New York']].dropna()
    data = temp_NY['New York'].values
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    x = []
    y = []
    for i in range(sequence_length, len(data_normalized)):
        x.append(data_normalized[i - sequence_length:i])
        y.append(data_normalized[i, 0])
    x, y = np.array(x), np.array(y)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1), scaler



def create_dataloaders(x, y, batch_size):
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def plot_predictions(model, device, test_loader, scaler):
    model.eval()
    predicted_temps = []
    actual_temps = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data).cpu().numpy()
            predicted_temps.extend(output)
            actual_temps.extend(target.cpu().numpy())

    predicted_temps = np.array(predicted_temps).reshape(-1, 1)
    actual_temps = np.array(actual_temps).reshape(-1, 1)
    predicted_temps_transformed = scaler.inverse_transform(predicted_temps)
    actual_temps_transformed = scaler.inverse_transform(actual_temps)
    plt.figure(figsize=(10, 6))
    plt.plot(actual_temps_transformed, color='blue', label='Actual New York Temperature')
    plt.plot(predicted_temps_transformed, color='red', linestyle='--', label='Predicted New York Temperature')
    plt.title('New York Temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PyTorch RNN for NY Temperature Prediction')
    parser.add_argument('--sequence-length', type=int, default=50,
                        help='sequence length for input data (default: 50)')
    parser.add_argument('--input-dim', type=int, default=1,
                        help='input dimension size (default: 5)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='number of hidden units in RNN (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of RNN layers (default: 2)')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='output dimension (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables MPS training (for macOS with Apple Silicon)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    # if use_cuda:
    #     device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    # print(f"Using device: {device}")

    x, y, scaler = load_and_process_data("Data/temperature.csv", args.sequence_length)
    split_idx = int(len(x) * 0.2)
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]

    train_loader = create_dataloaders(x_train, y_train, args.batch_size)
    test_loader = create_dataloaders(x_test, y_test, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RNNNet(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    plot_predictions(model, device, test_loader, scaler)

    if args.save_model:
        torch.save(model.state_dict(), "ny_temperature_rnn.pt")

if __name__ == "__main__":
    main()