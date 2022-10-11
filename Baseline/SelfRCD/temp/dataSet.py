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


class CDMData(HeteroData):
    def __init__(self):
        super().__init__()

    def __inc__(self, key: str, value, store=None, *args, **kwargs):
        if 'track' in key:
            return int(value.max()) + 1
        else:
            return 0


class RCD_DataSet(DataSet):
    def __init__(self, datadir, concept_graph_dir, dataSetName):
        super(RCD_DataSet, self).__init__(datadir, dataSetName)
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

        