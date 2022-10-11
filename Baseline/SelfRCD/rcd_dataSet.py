import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
sys.path.append("../..")
from initial_dataSet import DataSet


class RCDData(HeteroData):
    def __init__(self):
        super().__init__()

    def __inc__(self, key: str, value, store=None, *args, **kwargs):
        if 'track' in key:
            return int(value.max()) + 1
        else:
            return 0


class RCD_DataSet(DataSet):
    def __init__(self, datadir, concept_graph_dir, dataSetName):
        super(RCD_DataSet, self).__init__(datadir, dataSetName, build=True)
        self.datadir = datadir
        self.dataSetName = dataSetName

        if dataSetName == 'ASSIST_0910':
            graph_dir = concept_graph_dir + 'a0910/concept_relationship.csv'
        elif dataSetName == 'ASSIST_2017':
            graph_dir = concept_graph_dir + 'a2017/concept_relationship.csv'
        elif dataSetName == 'MAT_2016':
            graph_dir = concept_graph_dir + 'mat2016/concept_relationship.csv'
        elif dataSetName == 'JUNYI':
            graph_dir = concept_graph_dir + 'junyi/concept_relationship.csv'
        elif dataSetName == 'FrcSub':
            graph_dir = concept_graph_dir + 'frcsub/concept_relationship.csv'
        elif dataSetName == 'MathEC':
            graph_dir = concept_graph_dir + 'math_ec/concept_relationship.csv'
        self.graph_dir = graph_dir

        self.all_graph = self.get_all_graph()

    def get_concept_df(self) -> pd.DataFrame:
        concept_graph_df = pd.read_csv(self.graph_dir)
        return concept_graph_df.astype('int')

    def get_item_concept_df(self) -> pd.DataFrame:
        item = self.item
        item = item[~item.index.duplicated()]  # 去除重复项
        item_list, concept_list = [], []
        for idx in item.index:
            now_concept_list = eval(item.loc[idx, 'knowledge_code'])
            item_list.extend([idx] * len(now_concept_list))
            concept_list.extend(now_concept_list)
        return pd.DataFrame({'item': item_list, 'concept': concept_list}).astype('int')

    def get_stu_item_df(self) -> pd.DataFrame:
        return self.record.reset_index()

    def get_all_graph(self) -> RCDData:
        item_conc_df = self.get_item_concept_df()
        stu_item_df = self.get_stu_item_df()
        conc_conc_df = self.get_concept_df()
        stu_list = stu_item_df['user_id'].unique()
        item_list = list(set(stu_item_df['item_id'].unique()) |
                         set(item_conc_df['item'].unique()))
        conc_list = list(set(item_conc_df['concept'].unique()) |
                         set(conc_conc_df['concept_1'].unique()) |
                         set(conc_conc_df['concept_2'].unique()))

        assert len(stu_list) == len(self.total_stu_list), "students num wrong!"
        self.item_conc_df = item_conc_df
        self.stu_item_df = stu_item_df
        self.conc_conc_df = conc_conc_df
        self.graph_stu_list = stu_list
        self.graph_item_list = item_list
        self.graph_conc_list = conc_list

        stu_maping, item_maping, conc_maping, data = self.get_graph(stu_item_df, item_conc_df, conc_conc_df,
                                                                    stu_list, item_list, conc_list, all=True)
        self.stu_maping = stu_maping
        self.item_maping = item_maping
        self.conc_maping = conc_maping

        return data

    def get_data_loader(self, record: pd.DataFrame, batch_size: int) -> list:
        if 'user_id' not in record.columns:
            record = record.reset_index()
        all_stu_list = record['user_id'].unique()
        index_loader = DataLoader(TensorDataset(torch.tensor(all_stu_list)),
                                  batch_size=1, shuffle=True)
        batch_data_list = []
        for batch_idxs in tqdm(index_loader, "[Graphing:]"):
            stu_list = np.array([x.numpy()for x in batch_idxs], dtype='int').reshape(-1)
            stu_item_df = record.set_index('user_id').loc[stu_list, :].reset_index()

            item_list = stu_item_df['item_id'].unique()
            item_conc_df = self.item_conc_df.set_index('item').loc[item_list, :].reset_index()

            conc_list = item_conc_df['concept'].unique()
            conc_conc_df = self.conc_conc_df

            stu_maping, item_maping, conc_maping, data = self.get_graph(stu_item_df, item_conc_df, conc_conc_df,
                                                                        stu_list, item_list, conc_list, all=False)

            data['stu'].pos = torch.LongTensor(np.vectorize(self.stu_maping.get)(stu_list))
            data['item'].pos = torch.LongTensor(np.vectorize(self.item_maping.get)(item_list))
            data['conc'].pos = torch.LongTensor(np.vectorize(self.conc_maping.get)(conc_list))

            data['stu', 'master', 'conc'].edge_index = get_stu_conc_edge_index(stu_list, conc_list, stu_maping, conc_maping)
            data.label = torch.tensor(stu_item_df['score'].to_numpy()).float()

            batch_data_list.append(data)

        data_loader = PyG_DataLoader(batch_data_list, batch_size=batch_size, shuffle=True)

        return data_loader

    def get_graph(self, stu_item_df, item_conc_df, conc_conc_df,
                  stu_list, item_list, conc_list, all=True):

        stu_maping, item_maping, conc_maping = [dict(zip(x, range(len(x))))
                                                for x in [stu_list, item_list, conc_list]]
        data = RCDData()
        data['stu'].num_nodes = len(stu_list)
        data['item'].num_nodes = len(item_list)
        data['conc'].num_nodes = len(conc_list)

        stu_item_edge_index = get_edge_index(stu_item_df, ['user_id', 'item_id'], [stu_maping, item_maping])
        item_conc_edge_index = get_edge_index(item_conc_df, ['item', 'concept'], [item_maping, conc_maping])
        data['stu', 'answer', 'item'].edge_index = stu_item_edge_index
        data['item', 'contain', 'conc'].edge_index = item_conc_edge_index

        if all:
            conc_conc_edge_index = get_edge_index(conc_conc_df, ['concept_1', 'concept_2'], [conc_maping, conc_maping])
            data['conc', 'relevant', 'conc'].edge_index = conc_conc_edge_index
        else:
            stu_idxs, item_idxs, conc_idxs = [], [], []
            mean_idxs = []  # 最终聚合同学生同习题不同知识点的索引
            for idx in range(stu_item_edge_index.shape[1]):
                stu_idx = stu_item_edge_index[0][idx]
                item_idx = stu_item_edge_index[1][idx]
                conc_idx = item_conc_edge_index[1][item_conc_edge_index[0] == item_idx]
                stu_idxs.extend([stu_idx.item()] * len(conc_idx))
                item_idxs.extend([item_idx.item()] * len(conc_idx))
                conc_idxs.extend(conc_idx.tolist())
                mean_idxs.extend([idx] * len(conc_idx))

            data.stu_track = torch.LongTensor(stu_idxs)  # 为了在特征层取出包的嵌入
            data.item_index = torch.LongTensor(item_idxs)
            data.conc_index = torch.LongTensor(conc_idxs)
            data.mean_track = torch.LongTensor(mean_idxs)

        return stu_maping, item_maping, conc_maping, data


def get_edge_index(df, col_name, maping):
    # 创建从 j-->i 的图index
    # j : source  i : target
    j_name, i_name = col_name
    j_map, i_map = maping
    j_idx = list(df[j_name].map(j_map))
    i_idx = list(df[i_name].map(i_map))
    edge_index = torch.LongTensor([j_idx, i_idx])
    return edge_index


def get_stu_conc_edge_index(stu_list, conc_list, stu_map, conc_map):
    stus, concs = [], []
    for stu in stu_list:
        stus.extend([stu] * len(conc_list))
        concs.extend(conc_list)
    stus = list(np.vectorize(stu_map.get)(stus))
    concs = list(np.vectorize(conc_map.get)(concs))
    edge_index = torch.LongTensor([stus, concs])
    return edge_index


if __name__ == '__main__':
    # ----------基本参数--------------
    datadir = '../../'
    concept_graph_dir = './graph/'
    dataSet_list = ('FrcSub', 'Math1', 'Math2', 'ASSIST_0910', 'ASSIST_2017', 'MAT_2016', 'JUNYI')

    dataSet_idx = 0
    test_ratio = 0.8
    batch_size = 64
    learn_rate = 9e-2
    emb_dim = 16

    data_set_name = dataSet_list[dataSet_idx]
    epochs = 8
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = RCD_DataSet(datadir, concept_graph_dir, data_set_name)
    train, test = dataSet.get_train_test(dataSet.record, test_ratio=0.2)
    data_loader = dataSet.get_data_loader(train, batch_size=batch_size)
