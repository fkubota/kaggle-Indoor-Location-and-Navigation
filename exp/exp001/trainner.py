from ipdb import set_trace as st
from icecream import ic
import gc
import os
import wandb
import pandas as pd
from fastprogress import progress_bar
from loguru import logger
import numpy as np
import torch
from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import utils as U
import configuration as C
import result_handler as rh
# from criterion import mixup_criterion
# from early_stopping import EarlyStopping

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim

def train_cv(config, run_name):
    # config
    debug = config['globals']['debug']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fold = config['split']['n_fold']
    n_epoch = config['globals']['num_epochs']
    path_trn_tp = config['path']['path_train_tp']
    n_classes = config['model']['params']['n_classes']
    dir_save_exp, dir_save_ignore_exp, exp_name = U.get_save_dir_exp(
                                                            config, run_name)


    # ----------------
    df_train, df_test, sub, bssid_feats, rssi_feats, wifi_bssids_size = get_dataset(config)


    oofs = []  # 全てのoofをdfで格納する
    predictions = []  # 全ての予測値をdfで格納する
    val_scores = []
    # skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=n_fold)
    # for fold, (trn_idx, val_idx) in enumerate(skf.split(train.loc[:, 'path'], train.loc[:, 'path'])):
    for i_fold, (trn_idx, val_idx) in enumerate(gkf.split(df_train.loc[:, 'path'], groups=df_train.loc[:, 'path'])):

        # 指定したfoldのみループを回す
        print('=' * 20)
        print(f'Fold {i_fold+1}')
        print('=' * 20)

        # train/valid data
        trn_df = df_train.loc[trn_idx, bssid_feats + rssi_feats + ['x','y','floor']].reset_index(drop=True)
        val_df = df_train.loc[val_idx, bssid_feats + rssi_feats + ['x','y','floor']].reset_index(drop=True)

        # data loader
        loaders = {}
        loader_config = config["loader"]
        loaders["train"] = DataLoader(IndoorDataset(trn_df, 'train', bssid_feats, rssi_feats), **loader_config["train"], worker_init_fn=worker_init_fn) 
        loaders["valid"] = DataLoader(IndoorDataset(val_df, 'valid', bssid_feats, rssi_feats), **loader_config["valid"], worker_init_fn=worker_init_fn)
        loaders["test"] = DataLoader(IndoorDataset(df_test, 'test', bssid_feats, rssi_feats), **loader_config["test"], worker_init_fn=worker_init_fn)
        
        # model
        model = LSTMModel(wifi_bssids_size, num_feats=config['globals']['num_feats'])
        model_name = model.__class__.__name__
        
        # loggers
        wandb.init(project='kaggle-indoor',
                             group=f'{exp_name}_{run_name}',
                             name=f'fold{i_fold+1}')
        # wb_fold.config.config = config
        # wandb.init(project='Indoor_Location_Navigation', entity='sqrt4kaido', group=RUN_NAME, job_type=RUN_NAME + f'-fold-{i_fold+1}')
        # wandb.run.name = RUN_NAME + f'-fold-{i_fold+1}'
        wandb_config = wandb.config
        wandb_config.model_name = model_name
        wandb.watch(model)
        
        
        loggers = []
        loggers.append(WandbLogger())

        learner = Learner(model, config)
        
        # callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            monitor=f'Loss/val',
            mode='min',
            dirpath=dir_save_ignore_exp,
            verbose=False,
            filename=f'{model_name}-{learner.current_epoch}-{i_fold+1}')
        callbacks.append(checkpoint_callback)

        early_stop_callback = EarlyStopping(
            monitor='Loss/val',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='min')
        callbacks.append(early_stop_callback)
        
        trainer = pl.Trainer(
            logger=loggers,
            checkpoint_callback=callbacks,
            max_epochs=config['globals']['num_epochs'],
            default_root_dir=dir_save_ignore_exp,
            gpus=1,
            # fast_dev_run=config['globals']['debug'],
            deterministic=True,
            benchmark=True,
            )


        trainer.fit(learner, train_dataloader=loaders['train'], val_dataloaders=loaders['valid'])

        #############
        # validation (to make oof)
        #############
        model.eval()
        oof_x, oof_y, oof_f = evaluate(model, loaders, phase="valid")
        val_df["oof_x"] = oof_x
        val_df["oof_y"] = oof_y
        val_df["oof_floor"] = oof_f
        oofs.append(val_df)
        
        val_score = mean_position_error(
            val_df["oof_x"].values, val_df["oof_y"].values, 0,
            val_df['x'].values, val_df['y'].values, 0)
        val_scores.append(val_score)
        print(f"fold {i_fold+1}: mean position error {val_score}")

        #############
        # inference
        #############
        preds_x, preds_y, preds_f = evaluate(model, loaders, phase="test")
        test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
        test_preds.columns = sub.columns
        test_preds["site_path_timestamp"] = df_test["site_path_timestamp"]
        test_preds["floor"] = test_preds["floor"].astype(int)
        predictions.append(test_preds)
        

def get_dataset(config):
    bssid_feats = [f'bssid_{i}' for i in range(config['globals']['num_feats'])]
    rssi_feats  = [f'rssi_{i}' for i in range(config['globals']['num_feats'])]

    with open(f'./../../data_ignore/input/wifi/train_all.pkl', 'rb') as f:
        # df_train = pickle.load(f)
        df_train = pickle.load(f).iloc[::100, :]
    with open(f'./../../data_ignore/input/wifi/test_all.pkl', 'rb') as f:
        df_test = pickle.load(f).iloc[::100, :]
    sub = pd.read_csv(config['path']['path_sample_submission'], index_col=0)

    # bssidの一覧作成
    # wifi_bassidにはtrainとtest両方のbssidの一覧が含まれる
    wifi_bssids = []
    for i in range(config['globals']['num_feats']):
        wifi_bssids.extend(df_train.iloc[:,i].values.tolist())
    wifi_bssids = list(set(wifi_bssids))

    wifi_bssids_size = len(wifi_bssids)
    logger.info(f'BSSID TYPES: {wifi_bssids_size}')

    wifi_bssids_test = []
    for i in range(config['globals']['num_feats']):
        wifi_bssids_test.extend(df_test.iloc[:,i].values.tolist())
    wifi_bssids_test = list(set(wifi_bssids_test))

    wifi_bssids_size = len(wifi_bssids_test)
    logger.info(f'BSSID TYPES: {wifi_bssids_size}')

    wifi_bssids.extend(wifi_bssids_test)
    wifi_bssids_size = len(wifi_bssids)
    logger.info(f'BSSID TYPES: {wifi_bssids_size}')

    # LabelEncoding & StandardScaler
    le = LabelEncoder()
    le.fit(wifi_bssids)
    le_site = LabelEncoder()
    le_site.fit(df_train['site_id'])

    ss = StandardScaler()
    ss.fit(df_train.loc[:, rssi_feats])

    df_train.loc[:, rssi_feats] = ss.transform(df_train.loc[:, rssi_feats])
    df_test.loc[:, rssi_feats] = ss.transform(df_test.loc[:, rssi_feats])
    for feat in bssid_feats:
        df_train.loc[:, feat] = le.transform(df_train.loc[:, feat])
        df_test.loc[:, feat] = le.transform(df_test.loc[:, feat])
        
        df_train.loc[:, feat] = df_train.loc[:, feat] + 1
        df_test.loc[:, feat] = df_test.loc[:, feat] + 1
        
    df_train.loc[:, 'site_id'] = le_site.transform(df_train.loc[:, 'site_id'])
    df_test.loc[:, 'site_id'] = le_site.transform(df_test.loc[:, 'site_id'])

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    return df_train, df_test, sub, bssid_feats, rssi_feats, wifi_bssids_size

# dataset
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
            'bssid_feats':self.bssid_feats[idx],
            'rssi_feats':self.rssi_feats[idx],
#             'site_id':self.site_id[idx]
        }
        if self.phase in ['train', 'valid']:
            target = {
                'xy':self.xy[idx],
                'floor':self.floor[idx]
            }
        else:
            target = {}
        return feature, target


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)



class LSTMModel(nn.Module):
    def __init__(self, bssid_size=94248, site_size=24, embedding_dim=64, num_feats=20):
        super(LSTMModel, self).__init__()
        
        # bssid
        # ->64次元に圧縮後sequence化にする
        # wifi_bssids_sizeが辞書の数を表す
        self.bssid_embedding = nn.Embedding(bssid_size, 64, max_norm=True)
        # site
        # ->2次元に圧縮後sequence化する
        # site_countが辞書の数を表す       
        self.site_embedding = nn.Embedding(site_size, 64, max_norm=True)

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
        self.linear1 = nn.Linear(in_features=concat_size, out_features=256)#, bias=False)
        self.bn2 = nn.BatchNorm1d(256)

        self.batch_norm1 = nn.BatchNorm1d(1)
        self.lstm1 = nn.LSTM(input_size=256,hidden_size=128,dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128,hidden_size=16,dropout=0.1, batch_first=True)

        self.fc_xy = nn.Linear(16, 2)
        # self.fc_x = nn.Linear(16, 1)
        # self.fc_y = nn.Linear(16, 1)
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

        # output [batch, 1, 1] -> [batch]
        # x_ = self.fc_x(x).view(-1)
        # y_ = self.fc_y(x).view(-1)
        xy = self.fc_xy(x).squeeze(1)
        floor = torch.relu(self.fc_floor(x)).view(-1)
        # return {"x":x_, "y":y_, "floor":floor} 
        return {"xy": xy, "floor": floor}


# Learner class(pytorch-lighting)
class Learner(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.xy_criterion = get_criterion(config)
        self.f_criterion = get_criterion(config)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.xy_criterion(output["xy"], y["xy"])
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        xy_loss = self.xy_criterion(output["xy"], y["xy"])
        f_loss = self.f_criterion(output["floor"], y["floor"])
        loss = xy_loss  # + f_loss
        mpe = mean_position_error(
            to_np(output['xy'][:, 0]), to_np(output['xy'][:, 1]), 0, 
            to_np(y['xy'][:, 0]), to_np(y['xy'][:, 1]), 0)
        
        # floor lossは現状は無視して良い
        self.log(f'Loss/val', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'Loss/xy', xy_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'Loss/floor', f_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'MPE/val', mpe, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return mpe
    
    def validation_epoch_end(self, outputs):
        avg_loss = np.mean(outputs)
        print(f'epoch = {self.current_epoch}, mpe_loss = {avg_loss}')

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Loss/val"}


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")
    base_optimizer_name = optimizer_config.get("base_name")
    optimizer_params = optimizer_config['params']

    if hasattr(optim, optimizer_name):
        optimizer = optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_params)
        return optimizer
    else:
        base_optimizer = optim.__getattribute__(base_optimizer_name)
        optimizer = globals().get(optimizer_name)(
            model.parameters(), 
            base_optimizer,
            **optimizer_config["params"])
        return  optimizer

def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get("params")
    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion = globals().get(loss_name)(**loss_params)

    return criterion


def mean_position_error(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat-x, 2) + np.power(yhat-y, 2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]


def to_np(input):
    return input.detach().cpu().numpy()


# oof
def evaluate(model, loaders, phase):
    x_list = []
    y_list = []
    f_list = []
    with torch.no_grad():
        for batch in loaders[phase]:
            x, y = batch
            output = model(x)
            x_list.append(to_np(output['xy'][:, 0]))
            y_list.append(to_np(output['xy'][:, 1]))
            f_list.append(to_np(output['floor']))

    x_list = np.concatenate(x_list)
    y_list = np.concatenate(y_list)
    f_list = np.concatenate(f_list)
    return x_list, y_list, f_list
