from ipdb import set_trace as st
import os
import yaml
import torch
import pickle
import random
import itertools
import subprocess
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


def init_exp(config, config_update, run_name):
    '''
    - git hashの取得
    - dir_saveの作成と、dir_saveの取得
    - configのupdate
    '''
    logger.info(':: in ::')

    # git の hash値を取得
    cmd = "git rev-parse --short HEAD"
    hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    logger.info(f'hash: {hash_}')

    # 保存ディレクトリの用意
    dir_save, dir_save_ignore, exp_name = get_save_dir_exp(config, run_name)
    logger.info(f'exp_name: {exp_name}_{run_name}')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    if not os.path.exists(dir_save_ignore):
        os.makedirs(dir_save_ignore)

    # configのupdateとconfig_updateの保存
    deepupdate(config, config_update)
    with open(f'{dir_save}/config_update.yml', 'w') as path:
        yaml.dump(config_update, path)
    logger.info(f'config_update: {config_update}')

    # set_seed
    set_seed(config['globals']['seed'])

    logger.info(':: out ::')
    return dir_save, dir_save_ignore, config


def deepupdate(dict_base, other):
    '''
    ディクショナリを再帰的に更新する
    ref: https://www.greptips.com/posts/1242/
    '''
    for k, v in other.items():
        if isinstance(v, dict) and k in dict_base:
            deepupdate(dict_base[k], v)
        else:
            dict_base[k] = v


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_save_dir_exp(config, run_name):
    _dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = _dir.split('/')[-1]
    dir_save_exp = f'{config["path"]["dir_save"]}{exp_name}/{run_name}'
    dir_save_ignore_exp = f'{config["path"]["dir_save_ignore"]}'\
                          f'{exp_name}/{run_name}'
    return dir_save_exp, dir_save_ignore_exp, exp_name


def get_debug_idx(trn_tp, trn_idxs, val_idxs, config):
    n_classes = config['model']['params']['n_classes']

    trn_tp_trn = trn_tp.iloc[trn_idxs].copy()
    trn_tp_val = trn_tp.iloc[val_idxs].copy()
    trn_tp_trn['idx_'] = trn_idxs
    trn_tp_val['idx_'] = val_idxs

    trn_idxs_debug = []
    val_idxs_debug = []
    for idx in range(n_classes):
        bools = trn_tp_trn.species_id == idx
        trn_idxs_debug.append(trn_tp_trn[bools]['idx_'].values[0])

        bools = trn_tp_val.species_id == idx
        val_idxs_debug.append(trn_tp_val[bools]['idx_'].values[0])

    return trn_idxs_debug, val_idxs_debug


def set_debug_config(config):
    if config['globals']['debug']:
        logger.info(':: debug mode ::')
        config['globals']['num_epochs'] = 2
        config['split']['n_fold'] = 2
        # config['loader']['train']['batch_size'] = 2
        # config['loader']['valid']['batch_size'] = 2
        # config['loader']['test']['batch_size'] = 2
        return config
    else:
        return config


def get_dataset(config):
    num_feats = config['feature']['num_feats']
    bssid_feats = [f'bssid_{i}' for i in range(num_feats)]
    rssi_feats = [f'rssi_{i}' for i in range(num_feats)]

    with open('./../../data_ignore/input/wifi/train_all.pkl', 'rb') as f:
        df_train = pickle.load(f)
        # df_train = pickle.load(f).iloc[::100, :]
    with open('./../../data_ignore/input/wifi/test_all.pkl', 'rb') as f:
        df_test = pickle.load(f)
        # df_test = pickle.load(f).iloc[::100, :]
    sub = pd.read_csv(config['path']['path_sample_submission'], index_col=0)

    # bssidの一覧作成
    # wifi_bassidにはtrainとtest両方のbssidの一覧が含まれる
    wifi_bssids = []
    for i in range(num_feats):
        wifi_bssids.extend(df_train.iloc[:, i].values.tolist())
    wifi_bssids = list(set(wifi_bssids))

    wifi_bssids_size = len(wifi_bssids)
    logger.info(f'BSSID TYPES: {wifi_bssids_size}')

    wifi_bssids_test = []
    for i in range(num_feats):
        wifi_bssids_test.extend(df_test.iloc[:, i].values.tolist())
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


def check_update_config_key(list_config_str):
    '''
    存在しないkeyをupdateしていないかチェック
    '''
    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(f'{pwd}/config.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    for i_run, config_str in enumerate(list_config_str, 1):
        config_update = yaml.safe_load(config_str)
        check_key(config, config_update)

    print('====== list_config_str checked OK =====')


def check_key(dict_base, other):
    for k, v in other.items():
        if isinstance(v, dict):
            assert k in dict_base.keys(), f'間違えたkeyを入力しています({k})'
            check_key(dict_base[k], v)
        else:
            assert k in dict_base.keys(), f'間違えたkeyを入力しています({k})'


def sec2time(sec):
    hour = int(sec//3600)
    minute = int((sec - 3600*hour)//60)
    second = int(sec - 3600*hour - 60*minute)

    hour = str(hour).zfill(2)
    minute = str(minute).zfill(2)
    second = str(second).zfill(2)
    str_time = f'{hour}:{minute}:{second}'

    return str_time


def get_dataset_fixed_bssid(config):
    num_feats = config['feature']['num_feats']
    # bssid_feats = [f'bssid_{i}' for i in range(num_feats)]
    fixed_bssid_feats = [f'fixed_bssid_{i}' for i in range(num_feats)]

    df_train = pd.read_csv('./../../data_ignore/nb/027/train_n_bixed_bssid_300.csv')
    df_test = pd.read_csv('./../../data_ignore/nb/027/test_n_bixed_bssid_300.csv')
    sub = pd.read_csv(config['path']['path_sample_submission'], index_col=0)

    # bssidの一覧作成
    # wifi_bassidにはtrainとtest両方のbssidの一覧が含まれる
    wifi_bssids = []
    for i in range(num_feats):
        wifi_bssids.extend(df_train.iloc[:, i].values.tolist())
    wifi_bssids = list(set(wifi_bssids))

    le_site = LabelEncoder()
    le_site.fit(df_train['site_id'])
    ss = StandardScaler()
    ss.fit(df_train.loc[:, fixed_bssid_feats])

    df_train.loc[:, fixed_bssid_feats] = ss.transform(
            df_train.loc[:, fixed_bssid_feats])
    df_test.loc[:, fixed_bssid_feats] = ss.transform(
            df_test.loc[:, fixed_bssid_feats])

    df_train.loc[:, 'site_id'] = le_site.transform(df_train.loc[:, 'site_id'])
    df_test.loc[:, 'site_id'] = le_site.transform(df_test.loc[:, 'site_id'])

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test, sub, fixed_bssid_feats
