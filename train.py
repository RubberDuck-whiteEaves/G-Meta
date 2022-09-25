import  torch, os
import  numpy as np
from    subgraph_data_processing import Subgraphs
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
# pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
# pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
# pickle序列化后的数据，可读性差，人一般无法识别。
import  random, sys, pickle
import  argparse

import networkx as nx
import numpy as np
from scipy.special import comb
from itertools import combinations 
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm
import dgl

from meta import Meta
import time
import copy
import psutil
from memory_profiler import memory_usage

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,0,1"

print(torch.version.cuda)
print(torch.__version__)



def collate(samples):
        graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

        return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx

def main():
    mem_usage = memory_usage(-1, interval=.5, timeout=1)
    # 为CPU设置随机种子为222
    torch.manual_seed(222)
    # 为所有GPU设置随机种子为222
    torch.cuda.manual_seed_all(222)

    np.random.seed(222)

    print(args)
    
    # root表示数据所在目录
    root = args.data_dir

    # load() 和 save() 函数是读写文件数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npy 的文件中。
    # allow_pickle：一个布尔值，如果为True，则使用Python pickle
    feat = np.load(root + 'features.npy', allow_pickle = True)

    with open(root + '/graph_dgl.pkl', 'rb') as f:
        # 反序列化对象。将文件中的数据解析为一个Python对象。
        # 在load(file)的时候，要让python能够找到类的定义，否则会报错
        # dgl_graph类型是list，其中的元素类型是dgl.graph.DGLGraph
        dgl_graph = pickle.load(f)

    if args.task_setup == 'Disjoint':    
        with open(root + 'label.pkl', 'rb') as f:
            # info类型是字典dict，存储了节点的标签
            info = pickle.load(f)
    elif args.task_setup == 'Shared':
        if args.task_mode == 'True':
            root = root + '/task' + str(args.task_n) + '/'
        with open(root + 'label.pkl', 'rb') as f:
            info = pickle.load(f)

    # 计算共有多少种label
    total_class = len(np.unique(np.array(list(info.values()))))
    print('There are {} classes '.format(total_class))

    if args.task_setup == 'Disjoint':
        labels_num = args.n_way
    elif args.task_setup == 'Shared':
        labels_num = total_class

    # 单个图才会进入这个循环
    if len(feat.shape) == 2:
        # single graph, to make it compatible to multiple graph retrieval.
        feat = [feat]    

    # config是在搭建GCN的模型各层的维度
    # [('GraphConv',[单个节点的特征数量,args.hidden_dim])]
    config = [('GraphConv', [feat[0].shape[1], args.hidden_dim])]

    if args.h > 1:
        config = config + [('GraphConv', [args.hidden_dim, args.hidden_dim])] * (args.h - 1)

    config = config + [('Linear', [args.hidden_dim, labels_num])]

    if args.link_pred_mode == 'True':
        config.append(('LinkPred', [True]))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 函数to的作用是原地(in-place)修改Module，它可以当成三种函数来使用：
    # function:: to(device=None, dtype=None, non_blocking=False)
    # function:: to(dtype, non_blocking=False)
    # function:: to(tensor, non_blocking=False)
    maml = Meta(args, config).to(device)


    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

    # 对于函数parameters，我们可以使用for param in model.parameters()来遍历网络模型中的参数，因为该函数返回的是一个迭代器iterator。我们在使用优化算法的时候就是将model.parameters()传给优化器Optimizer。与之类似的还有函数buffers、函数children和函数modules。

    # requires_grad函数返回模型中的参数是否需要计算梯度，默认值为True，若有parameter更新为False，则进行梯度计算时符合链式法则
    # 在训练时如果想要固定网络的底层，那么可以令这部分网络对应子图的参数requires_grad为False。这样，在反向过程中就不会计算这些参数对应的梯度
    

    # 以下两句话的语义在计算所有可训练的参数有多少个
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    
    max_acc = 0

    # —–深复制(即copy.deepcopy)，即将被复制对象完全再复制一遍作为独立的新个体单独存在。所以改变原有被复制对象不会对已经复制出来的新对象产生影响。 
    # —–而赋值操作(即=)，并不会产生一个独立的对象单独存在，他只是将原有的数据块打上一个新标签，所以当其中一个标签被改变的时候，数据块就会发生变化，另一个标签也会随之改变。
    model_max = copy.deepcopy(maml)
    
    db_train = Subgraphs(root, 'train', info, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, batchsz=args.batchsz, args = args, adjs = dgl_graph, h = args.h)
    db_val = Subgraphs(root, 'val', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph, h = args.h)
    db_test = Subgraphs(root, 'test', info, n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=100, args = args, adjs = dgl_graph, h = args.h)
    print('------ Start Training ------')
    s_start = time.time()
    max_memory = 0
    for epoch in range(args.epoch):
        # 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
        # 参数：
        # dataset (Dataset) – 加载数据的数据集。
        # batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
        # shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
        # sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
        # num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
        # collate_fn (callable, optional) –
        # pin_memory (bool, optional) –
        # drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

        # 这里设置的batch_size=args.task_num，由于db_train的__getitem__返回的某个任务的所有数据，所以db中包含的是args.task_num个任务的数据作为一个batch
        db = DataLoader(db_train, args.task_num, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
        s_f = time.time()
        # x_spt：存储了batch_size(=args.task_num)个任务n_ways k_shots个子图
        # y_spt：存储了batch_size(=args.task_num)个任务中心节点的label
        # c_spt：存储了batch_size(=args.task_num)个任务的所有中心节点id对应的index
        # n_spt：存储了batch_size(=args.task_num)个任务的n_ways k_shots个子图上的所有节点id
        # g_spt：存储了batch_size(=args.task_num)个任务的n_ways k_shots个子图对应的图id
        for step, (x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry) in enumerate(db):
            nodes_len = 0
            if step >= 1:
                data_loading_time = time.time() - s_r
            else:
                data_loading_time = time.time() - s_f
            s = time.time()
            # x_spt: a list of #task_num tasks, where each task is a mini-batch of k-shot * n_way subgraphs
            # y_spt: a list of #task_num lists of labels. Each list is of length k-shot * n_way int.                
            nodes_len += sum([sum([len(j) for j in i]) for i in n_spt])
            accs = maml(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
            max_memory = max(max_memory, float(psutil.virtual_memory().used/(1024**3)))
            if step % args.train_result_report_steps == 0:
                print('Epoch:', epoch + 1, ' Step:', step, ' training acc:', str(accs[-1])[:5], ' time elapsed:', str(time.time() - s)[:5], ' data loading takes:', str(data_loading_time)[:5], ' Memory usage:', str(float(psutil.virtual_memory().used/(1024**3)))[:5])
            s_r = time.time()
            
        # validation per epoch
        db_v = DataLoader(db_val, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_v:

            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
            accs_all_test.append(accs)

        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Epoch:', epoch + 1, ' Val acc:', str(accs[-1])[:5])
        if accs[-1] > max_acc:
            max_acc = accs[-1]
            model_max = copy.deepcopy(maml)
                
    db_t = DataLoader(db_test, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        accs_all_test.append(accs)

    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', str(accs[1])[:5])

    for x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry in db_t:
        accs = model_max.finetunning(x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat)
        accs_all_test.append(accs)
    
    #torch.save(model_max.state_dict(), './model.pt')

    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Early Stopped Test acc:', str(accs[-1])[:5])
    print('Total Time:', str(time.time() - s_start)[:5])
    print('Max Momory:', str(max_memory)[:5])
    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=3)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=24)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--input_dim', type=int, help='input feature dim', default=1)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim', default=64)
    argparser.add_argument('--attention_size', type=int, help='dim of attention_size', default=32)
    argparser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    argparser.add_argument("--no_finetune", default=True, type=str, required=False, help="no finetune mode.")
    argparser.add_argument("--task_setup", default='Disjoint', type=str, required=True, help="Select from Disjoint or Shared Setup. For Disjoint-Label, single/multiple graphs are both considered.")
    argparser.add_argument("--method", default='G-Meta', type=str, required=False, help="Use G-Meta")
    argparser.add_argument('--task_n', type=int, help='task number', default=1)
    argparser.add_argument("--task_mode", default='False', type=str, required=False, help="For Evaluating on Tasks")
    argparser.add_argument("--val_result_report_steps", default=100, type=int, required=False, help="validation report")
    argparser.add_argument("--train_result_report_steps", default=30, type=int, required=False, help="training report")
    argparser.add_argument("--num_workers", default=0, type=int, required=False, help="num of workers")
    argparser.add_argument("--batchsz", default=1000, type=int, required=False, help="batch size")
    argparser.add_argument("--link_pred_mode", default='False', type=str, required=False, help="For Link Prediction")
    argparser.add_argument("--h", default=2, type=int, required=False, help="neighborhood size")
    argparser.add_argument('--sample_nodes', type=int, help='sample nodes if above this number of nodes', default=1000)

    # tissue_PPI:多张图 同种label 
    args = argparser.parse_args([
        "--data_dir=/public/lhy/xhy/graduation_project/G-Meta/G-Meta_Data/tissue_PPI/",
        "--epoch=15",
        "--task_setup=Shared",
        "--task_mode=True",
        "--task_n=4",
        "--k_qry=10",
        "--k_spt=3",
        "--update_lr=0.01",
        "--update_step=10",
        "--meta_lr=5e-3",
        "--num_workers=0",
        "--train_result_report_steps=200",
        "--hidden_dim=128",
        "--task_num=4",
        "--batchsz=1000"])

    main()
