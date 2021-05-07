import numpy as np
from torch.utils.data import Dataset


class IndoorDataset(Dataset):
    def __init__(self, df, phase, bssid_feats, rssi_feats):
        self.df = df
        self.phase = phase
        self.bssid_feats = df[bssid_feats].values.astype(int)
        self.rssi_feats = df[rssi_feats].values.astype(np.float32)
#         self.site_id = df['site_id'].values.astype(int)

        if phase in ['train', 'valid']:
            self.xy = df[['x', 'y']].values.astype(np.float32)
            self.floor = df['floor'].values.astype(np.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        feature = {
            'bssid_feats': self.bssid_feats[idx],
            'rssi_feats': self.rssi_feats[idx],
        }
        if self.phase in ['train', 'valid']:
            target = {
                'xy': self.xy[idx],
                'floor': self.floor[idx]
            }
        else:
            target = {}
        return feature, target


class IndoorDatasetFixedBssid(Dataset):
    def __init__(self, df, phase, fixed_bssid_feats):
        self.df = df
        self.phase = phase
        self.fixed_bssid_feats = df[fixed_bssid_feats].values.astype(np.float32)
        self.site_id = df['site_id'].values.astype(int)

        if phase in ['train', 'valid']:
            self.xy = df[['x', 'y']].values.astype(np.float32)
            self.floor = df['floor'].values.astype(np.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        feature = {
            'fixed_bssid_feats': self.fixed_bssid_feats[idx],
            'site_id': self.site_id[idx]
        }
        if self.phase in ['train', 'valid']:
            target = {
                'xy': self.xy[idx],
                'floor': self.floor[idx]
            }
        else:
            target = {}
        return feature, target


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
