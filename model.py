import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, conf):
        """
        Args:
            classes (int): Number of classes for classification
        """
        super(Net, self).__init__()

        self.lstm = nn.LSTM(conf.lstm_in, conf.lstm_out, conf.lstm_layers, dropout=conf.dropout)
        self.fc = nn.Linear(conf.lstm_out, 1)

    def forward(self, x_dict):
        """
        Args:
            x(tensor): Input sentence of size (batch size, c, h, w)
        Returns: Output Tensor of size (batch size, n_classes)
        """
        out = []
        for key in ['place', 'action', 'audio', 'cast']:
            x = x_dict[key]
            out.append(x)
        out = torch.cat(out, dim=-1)
        # out = torch.cat((out, x_dict['shot_end_frame'].unsqueeze(-1)), dim=-1)
        # out = torch.cat((out, x_dict['genre'].unsqueeze(0).repeat(out.size(0), init_size, 1)), dim=-1)
        out, _ = self.lstm(out)
        out = self.fc(out[0])
        return out[1:].T


if __name__ == "__main__":
    print(torch.__version__)
