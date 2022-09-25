import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Classifier
from    copy import deepcopy
import  os
import  datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 求欧几里得距离
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def proto_loss_spt(logits, y_t, n_support):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')
    
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]   
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes

def proto_loss_qry(logits, y_t, prototypes):
    target_cpu = y_t.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0]/n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val

# 定义整个meta模型的类
# 定义自已的网络：
#     需要继承nn.Module类，并实现forward方法。
#     一般把网络中具有可学习参数的层放在构造函数__init__()中，
#     不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr # float: 内循环的 learning rate
        self.meta_lr = args.meta_lr #外循环的learning rate
        self.n_way = args.n_way # disjoint label才有效，指一个task需要预测的label数量
        self.k_spt = args.k_spt # support shots，每个task中support set的中心节点为某种label的数量
        self.k_qry = args.k_qry # query shots
        self.task_num = args.task_num   # task数量
        self.update_step = args.update_step # update steps
        self.update_step_test = args.update_step_test   # fine-tuning是微调的意思，是用别人训练好的模型(即pre-trained model)，加上我们自己的数据，来训练新的模型。fine tune相当于使用别人的模型的前几层，来提取浅层特征，然后在最后再落入我们自己的分类中。

        starttime = datetime.datetime.now()
        m=torch.tensor([1,2,3])
        print(1111111)
        m.to(device)
        print(123123123)
        endtime = datetime.datetime.now()
        print (endtime - starttime)

        self.net = Classifier(config)
        starttime = datetime.datetime.now()
        
        self.net = self.net.to(device)
        endtime = datetime.datetime.now()
        print (endtime - starttime)
        
        # 设置Adam优化器
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.method = args.method

    def forward_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_spt)  #任务数量
        querysz = len(y_qry[0])  # 一个任务中query set中心节点的数量
        losses_s = [0 for _ in range(self.update_step)]
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        # i指batch_size中第i个任务
        for i in range(task_num):
            feat_spt = torch.Tensor(np.vstack(([feat[g_spt[i][j]][np.array(x)] for j, x in enumerate(n_spt[i])]))).to(device)  # 获取第i个任务中，support set对应的2*3张子图所有节点的feature（shape=[2095,50]）（子图之间可能有节点重复，不除重都放在其中）
            feat_qry = torch.Tensor(np.vstack(([feat[g_qry[i][j]][np.array(x)] for j, x in enumerate(n_qry[i])]))).to(device)
            # 1. run the i-th task and compute loss for k=0
            # 第一次更新使用sample的parameters
            logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, vars=None)  # logits存储一个任务中support set中的2*3张子图中心节点预测的label概率(未归一化)
            # y_spt[i]存储了任务i中心节点的label
            loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, self.net.parameters())  # 在forward函数里计算了loss，并进行反向传播
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  # 更新一次parameters

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                # 不进行梯度下降，直接测试query集
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, self.net.parameters())
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[0] += loss_q   # losses_q[0]存储测试集使用未进行更新的参数(未进行更新的参数实际上就是正态分布等随机方式得到的参数...)得到的loss
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # 进行一次梯度下降，测试query集
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q
            # k可以理解为epoch，每个training task不再存在batch的概念，或者可以理解一个batch就是2*3个子图，相当于整个support set
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits, _ = self.net(x_spt[i].to(device), c_spt[i].to(device), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(logits, y_spt[i], self.k_spt)
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), feat_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry(logits_q, y_qry[i], prototypes)
                losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        # task_num个任务的参数更新了k次后，累加各自最终query set上的loss和
        loss_q = losses_q[-1] / task_num
        
        if torch.isnan(loss_q):
            pass
        else:    
            # optimize theta parameters
            # 计算query set上的loss对parameters的梯度，并对net.parameters进行一次梯度更新(注意：之前的所有操作都没有对net.parameters进行梯度更新，仅仅只是用一个新的变量记录了更新的参数值)
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        accs = np.array(corrects) / (task_num)

        return accs

    def finetunning_ProtoMAML(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
        querysz = len(y_qry[0])

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        n_spt = n_spt[0]
        n_qry = n_qry[0]
        g_spt = g_spt[0]
        g_qry = g_qry[0]

        feat_spt = torch.Tensor(np.vstack(([feat[g_spt[j]][np.array(x)] for j, x in enumerate(n_spt)]))).to(device)
        feat_qry = torch.Tensor(np.vstack(([feat[g_qry[j]][np.array(x)] for j, x in enumerate(n_qry)]))).to(device)
            

        # 1. run the i-th task and compute loss for k=0
        logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt)
        loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, net.parameters())
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[0] = corrects[0] + acc_q
        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[1] = corrects[1] + acc_q


        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits, _ = net(x_spt.to(device), c_spt.to(device), feat_spt, fast_weights)
            loss, _, prototypes = proto_loss_spt(logits, y_spt, self.k_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q, _ = net(x_qry.to(device), c_qry.to(device), feat_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, acc_q = proto_loss_qry(logits_q, y_qry, prototypes)
            corrects[k + 1] = corrects[k + 1] + acc_q

        del net
        accs = np.array(corrects) 

        return accs

    # nn.Module类的__call__函数调用了forward函数，所以在直接调用Meta的对象时，会自动调用这个被重写的forward函数
    def forward(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        if self.method == 'G-Meta':
            accs = self.forward_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry,feat):
        if self.method == 'G-Meta':
            accs = self.finetunning_ProtoMAML(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        return accs
