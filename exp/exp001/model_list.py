from ipdb import set_trace as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LSTMModel(nn.Module):
    def __init__(self, bssid_size=94248, site_size=24,
                 embedding_dim=64, num_feats=20):
        super(LSTMModel, self).__init__()

        # bssid
        # ->64次元に圧縮後sequence化にする
        # wifi_bssids_sizeが辞書の数を表す
        self.bssid_embedding = nn.Embedding(bssid_size, embedding_dim, max_norm=True)
        # site
        # ->2次元に圧縮後sequence化する
        # site_countが辞書の数を表す
        self.site_embedding = nn.Embedding(site_size, embedding_dim, max_norm=True)

        # rssi
        # 次元を64倍に線形変換
        self.rssi = nn.Sequential(
            nn.BatchNorm1d(num_feats),
            nn.Linear(num_feats, num_feats * 64)
        )

        concat_size = (num_feats * 64) + (num_feats * 64)
        self.linear_layer2 = nn.Sequential(
            nn.BatchNorm1d(concat_size),
            nn.Dropout(0.3),
            nn.Linear(concat_size, 256),
            nn.ReLU()
        )
        self.bn1 = nn.BatchNorm1d(concat_size)

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(in_features=concat_size, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)

        self.batch_norm1 = nn.BatchNorm1d(1)
        self.lstm1 = nn.LSTM(
                input_size=256, hidden_size=128, dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(
                input_size=128, hidden_size=16, dropout=0.1, batch_first=True)

        self.fc_xy = nn.Linear(16, 2)
        self.fc_floor = nn.Linear(16, 1)

    def forward(self, x):
        # input embedding
        batch_size = x["bssid_feats"].shape[0]
        x_bssid = self.bssid_embedding(x['bssid_feats'])
        x_bssid = self.flatten(x_bssid)

#         x_site_id = self.site_embedding(x['site_id'])
#         x_site_id = self.flatten(x_site_id)

        x_rssi = self.rssi(x['rssi_feats'])

        x = torch.cat([x_bssid, x_rssi], dim=1)
        x = self.linear_layer2(x)

        # lstm layer
        x = x.view(batch_size, 1, -1)  # [batch, 1]->[batch, 1, 1]
        x = self.batch_norm1(x)
        x, _ = self.lstm1(x)
        x = torch.relu(x)
        x, _ = self.lstm2(x)
        x = torch.relu(x)

        xy = self.fc_xy(x).squeeze(1)
        floor = torch.relu(self.fc_floor(x)).view(-1)
        return {"xy": xy, "floor": floor}
