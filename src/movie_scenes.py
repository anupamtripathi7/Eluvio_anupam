import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import cosine_similarity_for_window
import pandas as pd
from tqdm import tqdm
import pickle
import glob
from src.utils import Config


root = '../'
split = 0.8
conf = Config()


class MovieScenes(Dataset):

    def __init__(self, window_size, transform=None):
        self.transform = transform
        # imdb_df = pd.read_csv(os.path.join(conf.data_path, 'title.basics.tsv'), sep='\t', index_col='tconst')
        # genre_idx = list(imdb_df.columns).index('genres')

        self.data = {'place': [],
                     'cast': [],
                     'action': [],
                     'audio': [],
                     'scene_transition_boundary_ground_truth': [],
                     'shot_end_frame': [],
                     'scene_transition_boundary_prediction': [],
                     'imdb_id': []}
        for n, file in tqdm(enumerate(glob.glob(os.path.join(conf.data_path, 'data/*.pkl')))):
            with open(file, 'rb') as f:
                pkl_data = pickle.load(f)
            for key, value in self.data.items():
                # if key == 'genre':
                #     self.data['genre'].append(torch.tensor(imdb_df.iloc[:, genre_idx+1:].loc[pkl_data['imdb_id']]))
                if key in ['place', 'cast', 'action', 'audio']:
                    self.data[key].append(cosine_similarity_for_window(pkl_data[key], window_size))
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
    transformed_dataset = MovieScenes(transform=None, window_size=conf.window_size)
    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)

    for sample in dataloader:
        print(sample['place'].size(), sample['action'].size())
