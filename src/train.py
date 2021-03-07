import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from src.model import Net
from src.movie_scenes import MovieScenes
from src.utils import get_accuracy, Config
from sklearn.metrics import classification_report, confusion_matrix


conf = Config()
conf.epochs = 5000
conf.lr = 5e-5
conf.window_size = 50
conf.lstm_layers = 4
conf.lstm_out = 32
conf.lstm_in = conf.window_size * 8
conf.train_test_split_ratio = 0.8
conf.dropout = 0.5
conf.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, valid_loader, load_model=False):
    """
    Train a model on the given dataset
    Args:
        train_loader (DataLoader): Train set data loader
        valid_loader (DataLoader): Validation set data loader
        load_model (bool): If true, loads a pre-trained model

    Returns:
        (nn.module): Trained model
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([11]).to(conf.device))
    model = Net(conf).to(conf.device)
    if load_model:
        model.load_state_dict(torch.load(os.path.join(conf.root, 'results', 'model.pth'), map_location=torch.device('cpu')))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    for epoch in tqdm(range(1, conf.epochs + 1)):
        train_loss, valid_loss = 0.0, 0.0
        y_true, y_pred, y_true_valid, y_pred_valid = [], [], [], []

        # Training
        model.train()
        for n, sample in enumerate(train_loader):
            optimizer.zero_grad()
            for key in ['place', 'action', 'audio', 'cast', 'scene_transition_boundary_ground_truth', 'shot_end_frame']:
                sample[key] = sample[key].to(conf.device)
            out = model(sample)
            loss = criterion(out, sample['scene_transition_boundary_ground_truth'].float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            sigmoid_out = torch.sigmoid(out)
            thresholded_out = (sigmoid_out > 0.5).float().detach().cpu()
            y_true.append(sample['scene_transition_boundary_ground_truth'].float().cpu())
            y_pred.append(thresholded_out)

        # Validate
        model.eval()
        for n, sample in enumerate(valid_loader):
            with torch.no_grad():
                inp_size = sample['place'].size(1)
                for key in ['place', 'action', 'audio', 'cast', 'scene_transition_boundary_ground_truth',
                            'shot_end_frame']:
                    sample[key] = sample[key].to(conf.device)
                out = model(sample)
                loss = criterion(out, sample['scene_transition_boundary_ground_truth'].float())
                valid_loss += loss.item()

                sigmoid_out = torch.sigmoid(out[:, -inp_size + 1:])
                thresholded_out = (sigmoid_out > 0.5).float().detach().cpu()
                y_true_valid.append(sample['scene_transition_boundary_ground_truth'].float().cpu())
                y_pred_valid.append(thresholded_out)

        if epoch % 25 == 0:
            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            y_true = torch.cat(y_true, dim=1)
            y_pred = torch.cat(y_pred, dim=1)
            y_true_valid = torch.cat(y_true_valid, dim=1)
            y_pred_valid = torch.cat(y_pred_valid, dim=1)
            accuracy = get_accuracy(y_true_valid, y_pred_valid)
            valid_accuracy = get_accuracy(y_true, y_pred)
            print('\n\nEpoch: {} \tTraining Loss: {:.6f} \tTrain accuracy: {}\t Valid accuracy: {}'.format(epoch,
                                                                                                           train_loss,
                                                                                                           accuracy,
                                                                                                           valid_accuracy))
            print(confusion_matrix(y_true.numpy().T, y_pred.numpy().T, labels=[0, 1]))
            print(classification_report(y_true.numpy().T, y_pred.numpy().T))
            print('##Valid##\n', confusion_matrix(y_true_valid.numpy().T, y_pred_valid.numpy().T, labels=[0, 1]))
            print(classification_report(y_true_valid.numpy().T, y_pred_valid.numpy().T), '\n\n\n')
        if epoch % 250 == 0:
            torch.save(model.state_dict(), os.path.join(conf.root, 'results', 'model.pth'))


if __name__ == '__main__':
    compose = None
    print("Extracting Dataset")
    transformed_dataset = MovieScenes(window_size=conf.window_size, transform=compose)
    print("Dataset loading done!\n")
    train_size = int(conf.train_test_split_ratio * len(transformed_dataset))
    train_set, valid_set = torch.utils.data.random_split(transformed_dataset,
                                                         [train_size, len(transformed_dataset) - train_size])
    train_load, valid_load = DataLoader(train_set, shuffle=False, batch_size=1), \
                                 DataLoader(valid_set, shuffle=False, batch_size=1)
    train(train_load, valid_load, True)
