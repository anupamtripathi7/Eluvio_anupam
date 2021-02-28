import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import pickle
import glob


data_path = 'data'
root = '../'
split = 0.8


class MovieScenes(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        imdb_df = pd.read_csv(os.path.join(data_path, 'title.basics.tsv'), sep='\t', index_col='tconst')
        genre_idx = list(imdb_df.columns).index('genres')

        self.data = {'place': [],
                     'cast': [],
                     'action': [],
                     'audio': [],
                     'scene_transition_boundary_ground_truth': [],
                     'shot_end_frame': [],
                     'scene_transition_boundary_prediction': [],
                     'imdb_id': [],
                     'genre': []}
        for file in tqdm(glob.glob(os.path.join(data_path, 'data/*.pkl'))):
            with open(file, 'rb') as f:
                pkl_data = pickle.load(f)
                for key, value in self.data.items():
                    if key == 'genre':
                        self.data['genre'].append(imdb_df.iloc[:, genre_idx+1:].loc[pkl_data['imdb_id']])
                    else:
                        self.data[key].append(pkl_data[key])

    def __len__(self):
        return len(self.data['place'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for key, value in self.data.items():
            sample[key] = value[idx]
        if self.transform:
            sample = self.transform()
        return sample


if __name__ == "__main__":
    transformed_dataset = MovieScenes(transform=None)
    dataloader = DataLoader(transformed_dataset, batch_size=5, shuffle=True)

    for sample in dataloader:
        print(sample['places'].size(), sample['action'].size())
