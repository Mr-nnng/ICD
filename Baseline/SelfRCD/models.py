import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from typing import Union, Tuple, Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch_scatter import scatter
from rcd_dataSet import RCD_DataSet


class GATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 bias: bool = True, linear_trans: bool = True, ** kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.linear_trans = linear_trans  # 是否将输入的x乘以权重矩阵

        if linear_trans:
            if isinstance(in_channels, int):
                self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
                self.lin_r = self.lin_l
            else:
                self.lin_l = Linear(in_channels[0], heads * out_channels, False)
                self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.linear_trans:
            glorot(self.lin_l.weight)
            glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, target_x, source_x, edge_index, size=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(target_x, Tensor) and isinstance(source_x, Tensor):
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            if self.linear_trans:
                x_l = self.lin_l(source_x).view(-1, H, C)
                x_r = self.lin_l(target_x).view(-1, H, C)
            else:
                x_l = source_x.view(-1, H, C)
                x_r = target_x.view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = target_x, source_x
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            if self.linear_trans:
                x_l = self.lin_l(x_l).view(-1, H, C)
            else:
                x_l = source_x.view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                if self.linear_trans:
                    x_r = self.lin_r(x_r).view(-1, H, C)
                else:
                    x_r = target_x.view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i,
                index: Tensor, ptr,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class Fusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Fusion, self).__init__()
        # concepts
        # source_target_fusion
        self.inter_conc_conc_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=True)  # p
        self.inter_item_conc_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=True)  # e
        self.exter_conc_conc_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=False)  # ηp
        self.exter_item_conc_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=False)  # ηe

        # items
        self.inter_conc_item_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=True)  # u
        self.inter_stu_item_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=True)   # v
        self.exter_conc_item_att = nn.Linear(2 * out_channels, 1, bias=False)  # γ1
        self.exter_stu_item_att = nn.Linear(2 * out_channels, 1, bias=False)   # γ2

        # students
        self.item_stu_fusion = GATConv(in_channels, out_channels, bias=False, linear_trans=True)   # q

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, data: HeteroData, data_x: dict):
        stu_num, item_num, conc_num = [data[name].num_nodes for name in ['stu', 'item', 'conc']]

        # ---------------concepts start-------------------------------
        # Intragraph fusion
        # row --> target_index , col --> source_index
        conc_conc_edge_index = SparseTensor(row=data['conc', 'relevant', 'conc'].edge_index[0],
                                            col=data['conc', 'relevant', 'conc'].edge_index[1],
                                            sparse_sizes=(conc_num, conc_num))
        target_conc_x = source_conc_x = data_x['conc']
        c_source_conc_inter = self.inter_conc_conc_fusion(target_x=target_conc_x, source_x=source_conc_x, edge_index=conc_conc_edge_index)

        item_conc_edge_index = SparseTensor(row=data['item', 'contain', 'conc'].edge_index[1],
                                            col=data['item', 'contain', 'conc'].edge_index[0],
                                            sparse_sizes=(conc_num, item_num))
        source_item_x = data_x['item']
        c_source_item_inter = self.inter_item_conc_fusion(target_x=target_conc_x, source_x=source_item_x, edge_index=item_conc_edge_index)

        # Extra graph fusion
        source_conc_conc_exter = self.exter_conc_conc_fusion(target_x=target_conc_x, source_x=c_source_conc_inter, edge_index=conc_conc_edge_index)
        source_item_conc_exter = self.exter_item_conc_fusion(target_x=target_conc_x, source_x=c_source_item_inter, edge_index=conc_conc_edge_index)
        conc_fused = target_conc_x + source_conc_conc_exter + source_item_conc_exter
        # ---------------concepts end-------------------------------

        # ----------------items start-------------------------------
        # Intragraph fusion
        conc_item_edge_index = SparseTensor(row=data['item', 'contain', 'conc'].edge_index[0],
                                            col=data['item', 'contain', 'conc'].edge_index[1],
                                            sparse_sizes=(item_num, conc_num))
        target_item_x, source_conc_x = data_x['item'], data_x['conc']
        i_source_conc_inter = self.inter_conc_item_fusion(target_x=target_item_x, source_x=source_conc_x, edge_index=conc_item_edge_index)

        stu_item_edge_index = SparseTensor(row=data['stu', 'answer', 'item'].edge_index[1],
                                           col=data['stu', 'answer', 'item'].edge_index[0],
                                           sparse_sizes=(item_num, stu_num))
        source_stu_x = data_x['stu']
        i_source_stu_inter = self.inter_stu_item_fusion(target_x=target_item_x, source_x=source_stu_x, edge_index=stu_item_edge_index)

        # Extra graph fusion

        source_conc_item_att = torch.softmax(self.exter_conc_item_att(torch.cat([i_source_conc_inter, target_item_x], dim=1)), dim=1)
        source_stu_item_att = torch.softmax(self.exter_stu_item_att(torch.cat([i_source_stu_inter, target_item_x], dim=1)), dim=1)
        item_fused = target_item_x + source_conc_item_att * i_source_conc_inter + source_stu_item_att * i_source_stu_inter
        # ----------------items end-------------------------------

        # ----------------students start--------------------------
        item_stu_edge_index = SparseTensor(row=data['stu', 'answer', 'item'].edge_index[0],
                                           col=data['stu', 'answer', 'item'].edge_index[1],
                                           sparse_sizes=(stu_num, item_num))
        target_stu_x = data['stu'].x
        s_source_item_inter = self.item_stu_fusion(target_x=target_stu_x, source_x=source_item_x, edge_index=item_stu_edge_index)
        stu_fused = target_stu_x + s_source_item_inter
        # ----------------students end--------------------------

        return conc_fused, item_fused, stu_fused


class Feature(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., bias: bool = True, ** kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(Feature, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = self.lin_l

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)

    def forward(self, target_x, source_x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        if isinstance(target_x, Tensor) and isinstance(source_x, Tensor):
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(source_x).view(-1, H, C)
            x_r = self.lin_r(target_x).view(-1, H, C)

        else:
            x_l, x_r = target_x, source_x
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        assert x_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        return torch.sigmoid(out)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        return x_j + x_i

    def aggregate(self, inputs: Tensor) -> Tensor:
        return inputs

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.lin_l.apply(clipper)
        self.lin_r.apply(clipper)


class Pred(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int = 1, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0., bias: bool = True, ** kwargs):
        super(Pred, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = self.lin_l

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)

    def forward(self, target_x, source_x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        if isinstance(target_x, Tensor) and isinstance(source_x, Tensor):
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(source_x).view(-1, H, C)
            x_r = self.lin_r(target_x).view(-1, H, C)

        else:
            x_l, x_r = target_x, source_x
            assert target_x.dim() == 2 and source_x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        assert x_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        return torch.sigmoid(out)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        return x_i - x_j

    def aggregate(self, inputs: Tensor) -> Tensor:
        return inputs

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.lin_l.apply(clipper)
        self.lin_r.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class Diagnosis(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super(Diagnosis, self).__init__()
        self.stu_feature = Feature(in_channels, in_channels)
        self.item_feature = Feature(in_channels, in_channels)
        self.pred = nn.Linear(in_channels, out_channels)
        self.cognitive_state_layer = nn.Linear(in_channels, out_channels)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, betch_data: HeteroData, return_cogn_start: bool = False):
        triple_index = (betch_data.stu_track, betch_data.item_index, betch_data.conc_index)
        mean_index = betch_data.mean_track
        stu_num, item_num, conc_num = [betch_data[name].num_nodes for name in ['stu', 'item', 'conc']]
        target_stu_x, source_conc_x = betch_data['stu'].x, betch_data['conc'].x
        conc_stu_edge_index = torch.cat((triple_index[2].view(1, -1), triple_index[0].view(1, -1)), dim=0)
        stu_feature = self.stu_feature(target_x=target_stu_x, source_x=source_conc_x, edge_index=conc_stu_edge_index)

        if return_cogn_start:
            cogn_state_input = scatter(stu_feature, triple_index[0], dim=0, reduce='mean')
            cognitive_state = torch.sigmoid(self.cognitive_state_layer(cogn_state_input))
            return cognitive_state.view(-1)

        conc_item_edge_index = SparseTensor(row=triple_index[1],
                                            col=triple_index[2],
                                            sparse_sizes=(item_num, conc_num))
        target_item_x = betch_data['item'].x
        item_feature = self.item_feature(target_x=target_item_x, source_x=source_conc_x, edge_index=conc_item_edge_index)

        pred_input = scatter(stu_feature - item_feature, mean_index, dim=0, reduce='mean')
        pred = torch.sigmoid(self.pred(pred_input))

        return pred.view(-1)

    def apply_clipper(self):
        self.stu_feature.apply_clipper()
        self.item_feature.apply_clipper()
        clipper = NoneNegClipper()
        self.pred.apply(clipper)


class RCD_Net(nn.Module):
    def __init__(self, stu_num, item_num, concept_num, in_channels: int, out_channels: int = 1):
        super(RCD_Net, self).__init__()
        self.stu_emb = nn.Parameter(torch.randn(stu_num, in_channels))
        self.item_emb = nn.Parameter(torch.randn(item_num, in_channels))
        self.conc_emb = nn.Parameter(torch.randn(concept_num, in_channels))

        self.fusion_1 = Fusion(in_channels, in_channels)
        self.fusion_2 = Fusion(in_channels, in_channels)
        self.diagnosis = Diagnosis(in_channels, out_channels)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, all_data, betch_data):
        data_x = {}
        data_x['conc'], data_x['item'], data_x['stu'] = [all_data[k].x for k in ['conc', 'item', 'stu']]
        data_x['conc'], data_x['item'], data_x['stu'] = self.fusion_1(all_data, data_x)
        conc_fused, item_fused, stu_fused = self.fusion_2(all_data, data_x)

        betch_data['conc'].x = conc_fused[betch_data['conc'].pos]
        betch_data['item'].x = item_fused[betch_data['item'].pos]
        betch_data['stu'].x = stu_fused[betch_data['stu'].pos]

        pred = self.diagnosis(betch_data)
        return pred

    def get_cogn_state(self, all_data, betch_data):
        data_x = {}
        data_x['conc'], data_x['item'], data_x['stu'] = [all_data[k].x for k in ['conc', 'item', 'stu']]
        data_x['conc'], data_x['item'], data_x['stu'] = self.fusion_1(all_data, data_x)
        conc_fused, item_fused, stu_fused = self.fusion_2(all_data, data_x)

        betch_data['conc'].x = conc_fused[betch_data['conc'].pos]
        betch_data['item'].x = item_fused[betch_data['item'].pos]
        betch_data['stu'].x = stu_fused[betch_data['stu'].pos]

        cogn_state = self.diagnosis(betch_data, return_cogn_start=True)
        return cogn_state

    def apply_clipper(self):
        self.diagnosis.apply_clipper()


class RCD():
    def __init__(self, stu_num, item_num, concept_num, emb_dim=8, learn_rate=1e-3, device='cpu'):
        self.stu_num = stu_num
        self.item_num = item_num
        self.concept_num = concept_num
        self.emb_dim = emb_dim
        self.device = device

        self.rcd_net = RCD_Net(stu_num, item_num, concept_num, emb_dim, 1).to(device)
        self.optimizer = torch.optim.Adam(self.rcd_net.parameters(), lr=learn_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.rcd_net.parameters())))

    def fit(self, all_data, data_loader, epochs, test_data=None):
        all_data['conc'].x, all_data['item'].x, all_data['stu'].x = \
            self.rcd_net.conc_emb, self.rcd_net.item_emb, self.rcd_net.stu_emb
        all_data = all_data.to(self.device)
        loss_list = []
        for epoch in range(epochs):
            for betch_data in tqdm(data_loader, "[Epoch:%s]" % (epoch + 1)):
                betch_data = betch_data.to(self.device)
                label = betch_data.label
                pred: Tensor = self.rcd_net(all_data, betch_data)
                loss: Tensor = self.loss_function(pred, label)
                # ------end training--------------------
                loss_list.append(loss.item())

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.rcd_net.apply_clipper()
                # ------ end update parameters-----------
            print('\t{} epoch loss = {}'.format(epoch + 1, np.mean(loss_list)))
            if test_data is not None:
                self.test(all_data, test_data)

    def test(self, all_data, data_loader):
        test_pred_list, test_label_list = [], []
        loss_list = []
        for betch_data in tqdm(data_loader, "[Testing:]"):
            betch_data = betch_data.to(self.device)
            with torch.no_grad():
                label = betch_data.label
                pred: Tensor = self.rcd_net(all_data, betch_data)
                loss: Tensor = self.loss_function(pred, label)
            loss_list.append(loss.item())
            test_pred_list.extend(pred.tolist())
            test_label_list.extend(label.tolist())
        print('\ttest loss = {}'.format(np.mean(loss_list)))
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


if __name__ == '__main__':

    # ----------基本参数--------------
    datadir = '../../'
    concept_graph_dir = './graph/'
    dataSet_list = ('FrcSub', 'ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'KDDCUP')

    dataSet_idx = 1
    test_ratio = 0.2
    batch_size = 32
    learn_rate = 9e-3

    data_set_name = dataSet_list[dataSet_idx]
    epochs = 80
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = RCD_DataSet(datadir, concept_graph_dir, data_set_name)

    train, test = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
    train_loader = dataSet.get_data_loader(train, batch_size=batch_size)
    test_loader = dataSet.get_data_loader(test, batch_size=batch_size)

    emb_dim = 16
    model = RCD(dataSet.student_num, dataSet.exercise_num, dataSet.concept_num, emb_dim=emb_dim, learn_rate=learn_rate, device=device)
    model.fit(dataSet.all_graph, train_loader, epochs=epochs, test_data=test_loader)
