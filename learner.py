import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
from torch.nn import init
import dgl
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,0,1"
# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# copied and editted from DGL Source 
# 定义图卷积结构的类
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = True
        self._activation = activation


    def forward(self, graph, feat, weight, bias):
        # local_var 会返回一个作用在内部函数中使用的Graph对象，对该对象的修改不会影响到原对象，属于对原对象的数据保护
        graph = graph.local_var()
        if self._norm:
            norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)  # 获得图中每个节点的度，并将小于1的度均设为1，最后对度tensor按元素求-0.5次方
            shp = norm.shape + (1,) * (feat.dim() - 1)  # 构造一个torch.size=[norm.shape,1]（1的数量取决于feat.dim() - 1）
            norm = torch.reshape(norm, shp).to(feat.device)  # 一维变两维，[norm.shape,]->[norm.shape,1]
            feat = feat * norm  # 相当于每个节点的50个特征乘该节点度的-0.5次方，即构造出pow(degree(j),-0.5)*hj

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            # 将节点i的所有邻居的pow(degree(j),-0.5)*hj聚合到i中，即构造出了∑pow(degree(j),-0.5)*hj
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            # 取出节点i刚刚得到的特征
            rst = graph.ndata['h']
            # 乘weight，即构造出了(∑pow(degree(j),-0.5)*hj)*W
            rst = torch.matmul(rst, weight)
        # 即构造出了(∑pow(degree(j),-0.5)*hj)*W*pow(degree(),-0.5)
        rst = rst * norm
        # 最后加上bias
        rst = rst + bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

# 定义网络结构的类
class Classifier(nn.Module):
    # 定义模型的各个层（可能嵌套了其他的自定义模型），每层的输入输出形状，以及weight and bias的shape和初始化
    def __init__(self, config):
        super(Classifier, self).__init__()
        
        #  参数列表
        self.vars = nn.ParameterList()
        self.graph_conv = []
        # 传入模型每层的参数形状
        self.config = config
        self.LinkPred_mode = False
        
        if self.config[-1][0] == 'LinkPred':
            self.LinkPred_mode = True
        
        # 初始化网络中所需要的所有参数
        for i, (name, param) in enumerate(self.config):
            
            if name is 'Linear':
                if self.LinkPred_mode:
                    w = nn.Parameter(torch.ones(param[1], param[0] * 2))
                else:
                    # w的shape：label种类*单个节点的特征数量
                    w = nn.Parameter(torch.ones(param[1], param[0]))
                # 以0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)，初始化w
                init.kaiming_normal_(w)
                self.vars.append(w)
                # bias
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            if name is 'GraphConv':
                # param: in_dim, hidden_dim
                w = nn.Parameter(torch.Tensor(param[0], param[1]))
                # torch.nn.init.xavier_uniform_(tensor, gain=1)
                # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
                # 这里有一个gain，增益的大小是依据激活函数类型来设定
                init.xavier_uniform_(w)
                self.vars.append(w)
                # bias
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.graph_conv.append(GraphConv(param[0], param[1], activation = F.relu))
            if name is 'Attention':
                # param[0] hidden size
                # param[1] attention_head_size
                # param[2] hidden_dim for classifier
                # param[3] n_ways
                # param[4] number of graphlets
                if self.LinkPred_mode:
                    w_q = nn.Parameter(torch.ones(param[1], param[0] * 2))
                else:
                    w_q = nn.Parameter(torch.ones(param[1], param[0]))
                w_k = nn.Parameter(torch.ones(param[1], param[0]))    
                w_v = nn.Parameter(torch.ones(param[1], param[4]))
                
                if self.LinkPred_mode:
                    w_l = nn.Parameter(torch.ones(param[3], param[2] * 2 + param[1]))
                else:
                    w_l = nn.Parameter(torch.ones(param[3], param[2] + param[1]))
                    
                init.kaiming_normal_(w_q)
                init.kaiming_normal_(w_k)
                init.kaiming_normal_(w_v)
                init.kaiming_normal_(w_l)

                self.vars.append(w_q)
                self.vars.append(w_k)
                self.vars.append(w_v)
                self.vars.append(w_l)

                #bias for attentions
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                #bias for classifier
                self.vars.append(nn.Parameter(torch.zeros(param[3])))

    # 进行模型内部的前向传播运算
    def forward(self, g, to_fetch, features, vars = None):
        # For undirected graphs, in_degree is the same as out_degree.

        if vars is None:
            vars = self.vars

        idx = 0 
        idx_gcn = 0

        h = features.float()
        h = h.to(device)

        for name, param in self.config:
            if name is 'GraphConv':
                w, b = vars[idx], vars[idx + 1]
                conv = self.graph_conv[idx_gcn]
                
                h = conv(g, h, w, b)  # 调用自定义GCN的前向传播运算

                g.ndata['h'] = h

                idx += 2 
                idx_gcn += 1

                if idx_gcn == len(self.graph_conv):
                    #h = dgl.mean_nodes(g, 'h')
                    num_nodes_ = g.batch_num_nodes
                    temp = [0] + num_nodes_
                    offset = torch.cumsum(torch.LongTensor(temp), dim = 0)[:-1].to(device)
                    
                    if self.LinkPred_mode:
                        h1 = h[to_fetch[:,0] + offset]
                        h2 = h[to_fetch[:,1] + offset]
                        h = torch.cat((h1, h2), 1)
                    else:
                        h = h[to_fetch + offset]    # 两层GCN得到了每个子图所有节点的特征，这里提取出需要预测label的节点特征
                        
            if name is 'Linear':
                w, b = vars[idx], vars[idx + 1]
                # nn.Linear和nn.functional.linear的区别：
                # 前者是类，要进行初始化，再调用其对象进行前向传播；后者是函数，直接调用即可进行前向传播
                # 前者在类初始化时，传入参数的形状，自动完成参数的初始化；后者需要手动初始化参数，再传入函数中
                # 举例：
                # m=nn.Linear(20,30)
                # input=torch.randn(128,20)
                # output=m(input)
                # or
                # input=torch.randn(128,20)
                # weight=nn.Parameter(torch.Tensor(30,20))
                # init.kaiming_normal_(weight)
                # bias=nn.Parameter(torch.zeros(30))
                # output=nn.functional.linear(input)

                # F.linear()参数的形状：
                # output=linear(input,weight,bias)
                # input:(N,*,in_features),*表示任意数量(>=0)的维度均可填充
                # weight:(out_features,in_features)
                # bias:(out_features)
                # output:(out_features,*,in_features)
                h = F.linear(h, w, b)
                idx += 2

            if name is 'Attention':
                w_q, w_k, w_v, w_l = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                b_q, b_k, b_v, b_l = vars[idx + 4], vars[idx + 5], vars[idx + 6], vars[idx + 7]

                Q = F.linear(h, w_q, b_q)
                K = F.linear(h_graphlets, w_k, b_k)

                attention_scores = torch.matmul(Q, K.T)
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                context = F.linear(attention_probs, w_v, b_v)
                
                # classify layer, first concatenate the context vector 
                # with the hidden dim of center nodes
                h = torch.cat((context, h), 1)
                h = F.linear(h, w_l, b_l)
                idx += 8
       
        return h, h
            
    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars