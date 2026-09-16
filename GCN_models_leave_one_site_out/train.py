import pandas as pd
from dataProc import MedicalDataset
from functions1 import run_leave_one_site_cv
import argparse


def load_dim(v, task):
    if task == 'ASD_multimodal':
        if v == 'RSFC_iRSSN':
            return 43 + 77
        elif v == 'RSFC_R2SN':
            return 22 + 77
        elif v == 'RSFC_BrainANet_CNN':
            return 64 + 77
        elif v == 'RSFC_BrainANet_SwinVIT':
            return 64 + 77
        elif v == 'RSFC':
            return 77
        elif v == 'R2SN':
            return 22
        elif v == 'BrainANet_CNN':
            return 64
        elif v == 'BrainANet_SwinVIT':
            return 64
        elif v == 'iRSSN':
            return 43

    elif task == 'MDD_multimodal':
        if v == 'RSFC_iRSSN':
            return 43 + 90
        elif v == 'RSFC_R2SN':
            return 22 + 90
        elif v == 'RSFC_BrainANet_CNN':
            return 64 + 90
        elif v == 'RSFC_BrainANet_SwinVIT':
            return 64 + 90
        elif v == 'RSFC':
            return 90
        elif v == 'R2SN':
            return 22
        elif v == 'BrainANet_CNN':
            return 64
        elif v == 'BrainANet_SwinVIT':
            return 64
        elif v == 'iRSSN':
            return 43
    else:
        if v == 'iRSSN':
            return 43
        elif v == 'R2SN':
            return 22
        elif v == 'BrainANet_CNN':
            return 64
        elif v == 'BrainANet_SwinVIT':
            return 64

    # elif v == 'RSFC_BrainANet_SwinVIT':
    #     return 64 + 192
    #
    # elif v == 'RSFC':
    #     return 192
    #
    # elif v == 'RSFC_R2SN':
    #     return 192 + 22
    #
    # elif v == 'RSFC_iRSSN':
    #     return 192 + 43
    # # elif v.startswith('CNN'):
    # #     return 32
    # # elif v.startswith('rest'):
    # #     return 90
    # else:
    #     return 128

def parse_str_or_int(value):
    try:
        # 尝试将输入转换为整数
        return int(value)
    except ValueError:
        # 如果转换失败，则保留为字符串
        return value
# metadata = pd.read_csv(r'DataTable_classification.csv')


def model_factory(model, net, n_node, task):
    if model == 'GAT':
        from models.GAT import GAT
        Net = GAT(input_dim=load_dim(net, task=task), hidden=128, n_node=n_node)

    elif model == 'GCN':
        from models.GCN import GCN
        Net = GCN(input_dim=load_dim(net, task=task), hidden=128)

    elif model == 'GIN':
        from models.GIN import GIN
        Net = GIN(input_dim=load_dim(net, task=task), hidden=128)

    elif model == 'GlobalAttentionNet':
        from models.global_attention import GlobalAttentionNet
        Net = GlobalAttentionNet(input_dim=load_dim(net, task=task), hidden=128)

    elif model == 'GraphSAGE':
        from models.graph_sage import GraphSAGE
        Net = GraphSAGE(input_dim=load_dim(net, task=task), hidden=128)

    # net.loss_dict = {
    #     'train_loss': [],
    #     'eval_loss': [],
    #     # 'train_loss_multi': {},
    #     # 'eval_loss_multi': {},
    # }

    return Net


# group_to_label = {
#     "NC": 0,
#     "AD": 1,
# }
def parse_str_or_int(value):
    try:
        # 尝试将输入转换为整数
        return int(value)
    except ValueError:
        # 如果转换失败，则保留为字符串
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default='R2SN',
                        help='BrainANet_CNN,BrainANet_SwinVIT,R2SN,RSFC,RSFC_BrainANet_SwinVIT')
    parser.add_argument('-model', default='GAT')
    # parser.add_argument('-g', default=['1', '2'], nargs='+', type=parse_str_or_int, help='Group')  # ['CN', 'AD']
    # parser.add_argument('-g', default=['1', '2'], nargs='+', help='Group', type=parse_str_or_int)  # ['CN', 'AD']
    parser.add_argument('-g', default=['1', '2'], nargs='+', help='Group')
    parser.add_argument('-task', default='ASD', help='Task(Dementia,ASD,MDD,ASD_multimodal,MDD_multimodal)')
    parser.add_argument('-table', default='DataTable_classification.csv')
    parser.add_argument('-seed', default=46, type=int)
    # /media/shucheng/数程SSD_2T/ADNI/derivatives/GretnaTimeCourse/data/ROIsignal_033_S_5259.csv
    args = parser.parse_args()

    dataset = MedicalDataset(
        dataframe=pd.read_csv(args.table), network=args.net, group=args.g, task=args.task
    )

    all_predictions, site_metrics = (
        run_leave_one_site_cv(
            metadata=dataset.dataframe,
            dataset=dataset,
            model_factory=model_factory,
            output_dir=f"./LOSO_results13/{args.net}.{args.model}.{args.task}.{args.g[0]}_{args.g[1]}.{args.seed}",
            n_node=90,
            num_epochs=100,
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=1e-4,
            val_ratio=0.25,
            num_workers=8,
            patience=20,
            seed=42,
            net_name=args.net,
            model_name=args.model,
            task=args.task,
        )
    )
