import torch
import torch.nn as nn
import os


class Config:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = '../'
        self.data_path = os.path.join(self.root, 'data')
        self.results_path = 'results'
        self.epochs = 5000
        self.lr = 5e-5
        self.window_size = 50
        self.lstm_layers = 4
        self.lstm_out = 32
        self.lstm_in = self.window_size * 8
        self.train_test_split_ratio = 0.8
        self.dropout = 0.5


def get_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).int().sum().item()
    return correct / y_true.shape[1]


def cosine_similarity_for_window(x, window_size=2):
    clips = x.size(0)
    cos = nn.CosineSimilarity(dim=-1)
    padded = torch.zeros(x.size(0)+window_size*2, x.size(1))
    padded[window_size:-window_size, :] = x
    out = []
    for w in range(-window_size, window_size+1):
        if w:
            mat = padded[window_size+w:window_size+clips+w, :]
            out.append(cos(x, mat))
    return torch.stack(out, dim=-1)