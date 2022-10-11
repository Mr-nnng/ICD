import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch_geometric.data import HeteroData


def get_data_info(record: pd.DataFrame):
    items = np.array(record.index).reshape(-1)
    scores = np.array(record['score']).reshape(-1)
    concepts = np.array(record['concept']).reshape(-1)
    i2c_idx = np.array(record['item_conc_index']).reshape(-1)

    item_unique, item_track = np.unique(items, return_inverse=True)
    conc_unique, conc_track = np.unique(concepts, return_inverse=True)
    unique_item_df = record.reset_index().drop_duplicates(subset='item_id', keep='first').set_index('item_id')
    unique_item_score = np.array(unique_item_df.loc[item_unique, 'score']).reshape(-1)
    return [item_unique, item_track], [conc_unique, conc_track], i2c_idx, [scores, unique_item_score]


def get_stu_df_list(extend_df, all_stu_list):
    stu_df_list = [extend_df.loc[x[0], :] for x in all_stu_list]
    return stu_df_list


def get_stu_data_list(pool_out):
    stu_data_list = []
    for i in pool_out:
        stu_data_list.extend(i)
    return stu_data_list


def get_extend_record(record, item_conc):
    extend_record = pd.merge(record.reset_index(), item_conc, left_on='item_id', right_on='item')\
        .drop(columns=['item']).set_index('user_id')
    return extend_record


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


class CDMData(HeteroData):
    def __init__(self):
        super().__init__()

    def __inc__(self, key: str, value, store=None, *args, **kwargs):
        if 'track' in key:
            return int(value.max()) + 1
        else:
            return 0


def get_data(source_df, target_df):
    data = CDMData()
    s_item, s_conc, s_i2c_idx, s_score = get_data_info(source_df)
    data['source'].unique_stu_index = torch.LongTensor(np.unique(source_df['user_id']) - 1)
    data['source'].unique_item_index = torch.LongTensor(s_item[0] - 1)
    data['source'].unique_conc_index = torch.LongTensor(s_conc[0] - 1)
    data['source'].item_track = torch.LongTensor(s_item[1])
    data['source'].conc_track = torch.LongTensor(s_conc[1])
    data['source'].stu_track = torch.zeros(len(s_item[0]), dtype=torch.long)
    data['source'].item_conc_index = torch.LongTensor(s_i2c_idx)
    data['source'].score = torch.FloatTensor(s_score[0])
    data['source'].unique_item_score = torch.FloatTensor(s_score[1])
    data['source'].stu_conc_track = torch.zeros(len(s_conc[0]), dtype=torch.long)

    t_item, t_conc, t_i2c_idx, t_score = get_data_info(target_df)
    data['target'].unique_item_index = torch.LongTensor(t_item[0] - 1)
    data['target'].unique_conc_index = torch.LongTensor(t_conc[0] - 1)
    data['target'].item_track = torch.LongTensor(t_item[1])
    data['target'].conc_track = torch.LongTensor(t_conc[1])
    data['target'].stu_track = torch.zeros(len(t_item[0]), dtype=torch.long)
    data['target'].item_conc_index = torch.LongTensor(t_i2c_idx)
    data['target'].score = torch.FloatTensor(t_score[0])
    data['target'].unique_item_score = torch.FloatTensor(t_score[1])
    data['target'].stu_conc_track = torch.zeros(len(t_conc[0]), dtype=torch.long)

    return data
