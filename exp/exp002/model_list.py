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


class FixedBssidMLP(nn.Module):
    def __init__(self, site_size=24, num_feats=20):
        super(FixedBssidMLP, self).__init__()

        self.flatten = nn.Flatten()
        self.site_embedding = nn.Embedding(site_size, 64, max_norm=True)
        self.fixed_bssid = nn.Sequential(
                    nn.BatchNorm1d(num_feats),
                    nn.Linear(num_feats, num_feats * 64)
        )

        concat_size = 64 + (num_feats * 64)
        self.bn1 = nn.BatchNorm1d(concat_size)

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(in_features=concat_size, out_features=256)#, bias=False)
        self.bn2 = nn.BatchNorm1d(256)

        self.linear2 = nn.Linear(in_features=256, out_features=128)#, bias=False)
        self.linear3 = nn.Linear(in_features=128, out_features=16)#, bias=False)

        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(16)

        self.xy = nn.Linear(in_features=16, out_features=2)#, bias=False)
        self.floor = nn.Linear(in_features=16, out_features=1)#, bias=False)

    # def forward(self, fixed_bssid, site):
    def forward(self, x):
        site = x['site_id']
        site = torch.reshape(site, (-1, 1))
        site_out = self.site_embedding(site)
        site_out = self.flatten(site_out)
        fixed_bssid = x['fixed_bssid_feats']
        fixed_bssid_out = self.fixed_bssid(fixed_bssid)

        x = torch.cat([site_out, fixed_bssid_out], dim=1)

        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.linear1(x))
        x = self.bn2(x)

        x = F.relu(self.linear2(x))
        x = self.bn3(x)

        x = F.relu(self.linear3(x))
        x = self.bn4(x)

        xy = self.xy(x).squeeze(1)
        floor = torch.relu(self.floor(x)).view(-1)
        return {"xy": xy, "floor": floor}
