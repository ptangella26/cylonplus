import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
import argparse



class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        final_output = self.fc(output)
        return final_output

def train(args, model, train_iterator, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (text, target) in enumerate(train_iterator):
        optimizer.zero_grad()
        output = model(text)
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_iterator),
                100. * batch_idx / len(train_iterator), loss.item()))


def test(model, test_iterator):
    model.eval()
    test_loss = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for text, target in test_iterator:
            output = model(text)
            output = output.view(-1, output.shape[-1])
            target = target.view(-1)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_tokens += target.size(0)

    test_loss /= total_tokens
    accuracy = 100. * correct / total_tokens

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_tokens, accuracy))


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Penn Treebank Language Modeling')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='size of word embeddings (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='number of hidden units in LSTM (default: 256)')
    parser.add_argument('--output-dim', type=int, default=10,
                        help='output dimension (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    tokenizer = get_tokenizer("basic_english")
    train_iter, valid_iter, test_iter = PennTreebank()

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    def collate_batch(batch):
        batch = [[vocab[token] for token in tokens] for tokens in batch]
        max_length = max(len(sequence) for sequence in batch)
        padded_sequences = torch.zeros(len(batch), max_length, dtype=torch.long)
        targets = torch.zeros(len(batch), max_length, dtype=torch.long)

        for i, sequence in enumerate(batch):
            length = len(sequence)
            padded_sequences[i, :length] = torch.tensor(sequence)
            targets[i, :length - 1] = torch.tensor(sequence[1:])

        return padded_sequences, targets

    train_data = list(yield_tokens(train_iter))
    test_data = list(yield_tokens(test_iter))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=True)
    test_batch_size = getattr(args, 'test_batch_size', 1000)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, collate_fn=collate_batch)

    model = LSTMModel(vocab_size=len(vocab), embedding_dim=128, hidden_dim=256, output_dim=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "ptb_lstm_model.pt")


if __name__ == '__main__':
    main()
