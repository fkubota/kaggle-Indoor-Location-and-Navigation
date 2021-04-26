from ipdb import set_trace as st
from icecream import ic
import gc
import wandb
import pandas as pd
from fastprogress import progress_bar
from loguru import logger
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import utils as U
import model_list as M
import configuration as C
import datasets as D
# import result_handler as rh
# from criterion import mixup_criterion
# from early_stopping import EarlyStopping

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
# import torch
from torch import nn
from pytorch_lightning.loggers import WandbLogger
import torch.optim as optim


def train_cv(config, run_name):
    # config
    debug = config['globals']['debug']
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_fold = config['split']['n_fold']
    n_epoch = config['globals']['num_epochs']
    num_feats = config['feature']['num_feats']
    dir_save_exp, dir_save_ignore_exp, exp_name = U.get_save_dir_exp(
                                                            config, run_name)

    # ----------------
    df_train, df_test, sub, bssid_feats, rssi_feats, wifi_bssids_size = U.get_dataset(config)

    oofs = []  # 全てのoofをdfで格納する
    predictions = []  # 全ての予測値をdfで格納する
    val_scores = []
    # skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=n_fold)
    # for fold, (trn_idx, val_idx) in enumerate(skf.split(train.loc[:, 'path'], train.loc[:, 'path'])):
    for i_fold, (trn_idx, val_idx) in enumerate(gkf.split(df_train.loc[:, 'path'], groups=df_train.loc[:, 'path'])):

        # 指定したfoldのみループを回す
        logger.info('=' * 20)
        logger.info(f'Fold {i_fold+1}')
        logger.info('=' * 20)

        # train/valid data
        trn_df = df_train.loc[trn_idx, bssid_feats + rssi_feats + ['x', 'y', 'floor']].reset_index(drop=True)
        val_df = df_train.loc[val_idx, bssid_feats + rssi_feats + ['x', 'y', 'floor']].reset_index(drop=True)
        if debug:
            trn_df = trn_df.iloc[::50, :]
            val_df = val_df.iloc[::50, :]

        # data loader
        loaders = {}
        loader_config = config["loader"]
        loaders["train"] = DataLoader(D.IndoorDataset(trn_df, 'train', bssid_feats, rssi_feats),
                                      **loader_config["train"], worker_init_fn=D.worker_init_fn)
        loaders["valid"] = DataLoader(D.IndoorDataset(val_df, 'valid', bssid_feats, rssi_feats),
                                      **loader_config["valid"], worker_init_fn=D.worker_init_fn)
        loaders["test"] = DataLoader(D.IndoorDataset(df_test, 'test', bssid_feats, rssi_feats),
                                     **loader_config["test"], worker_init_fn=D.worker_init_fn)

        # model
        model = M.LSTMModel(wifi_bssids_size, num_feats=num_feats)
        model_name = model.__class__.__name__

        # loggers
        wandb.init(project='kaggle-indoor',
                   group=f'{exp_name}_{run_name}',
                   name=f'fold{i_fold+1}')
        wandb_config = wandb.config
        wandb_config.model_name = model_name
        wandb_config.config = config
        wandb.watch(model)

        loggers = []
        loggers.append(WandbLogger())

        learner = Learner(model, config)
        
        # callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(
            monitor='Loss/val',
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
            max_epochs=n_epoch,
            default_root_dir=dir_save_ignore_exp,
            gpus=1,
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
        logger.info(f"fold {i_fold+1}: mean position error {val_score}")

        #############
        # inference
        #############
        preds_x, preds_y, preds_f = evaluate(model, loaders, phase="test")
        test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
        test_preds.columns = sub.columns
        test_preds["site_path_timestamp"] = df_test["site_path_timestamp"]
        test_preds["floor"] = test_preds["floor"].astype(int)
        predictions.append(test_preds)

        wandb.finish()

    ############
    # save oof #
    ############
    oofs_df = pd.concat(oofs)
    oofs_df.to_csv(f'{dir_save_exp}/oof.csv', index=False)

    ###################
    # save submission #
    ###################
    all_preds = pd.concat(predictions).groupby('site_path_timestamp').mean().reindex(sub.index)
    # floorの数値を置換
    simple_accurate_99 = pd.read_csv(config['path']['path_sample_submission_floor99'])
    all_preds['floor'] = simple_accurate_99['floor'].values
    all_preds.to_csv(f'{dir_save_exp}/sub.csv')

    ###########
    # summary #
    ###########
    wandb.init(project='kaggle-indoor',
               group=f'{exp_name}_{run_name}',
               name='summary')
    wandb.log({
        'CV_mean': np.mean(val_scores),
        'CV_std': np.std(val_scores)
        })
    wandb.finish()


# Learner class(pytorch-lighting)
class Learner(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.xy_criterion = C.get_criterion(config)
        self.f_criterion = C.get_criterion(config)

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
        self.log('Loss/val', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Loss/xy', xy_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('Loss/floor', f_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('MPE/val', mpe, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return mpe

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean(outputs)
        logger.info(f'epoch = {self.current_epoch}, mpe_loss = {avg_loss}')

    def configure_optimizers(self):
        optimizer = C.get_optimizer(self.model, self.config)
        scheduler = C.get_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Loss/val"}


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
