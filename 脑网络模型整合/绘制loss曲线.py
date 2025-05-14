import matplotlib.pyplot as plt
import pickle
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str,
                        default='/home/shucheng/python_files/深度学习模型/脑网络模型整合/result_save_param/SGCN_GAT.mask_0.2.SwinVIT.CN_AD/loss.pkl',
                        help='loss.pkl的绝对路径')
    parser.add_argument('-o', type=str, default='./loss_curve.png', help='保存图像的路径')
    args = parser.parse_args()

    if not os.path.isfile(args.i):
        raise ValueError(f'文件 {args.i}不存在！')

    # 加载损失数据
    loss_data = pickle.load(open(args.i, 'rb'))
    train_loss_data = loss_data['train_loss']
    eval_loss_data = loss_data['eval_loss']
    x = [i for i in range(len(train_loss_data))]

    # 设置图形样式
    plt.style.use('seaborn-darkgrid')  # 使用Seaborn的主题

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 绘制训练和验证损失曲线
    ax.plot(x, train_loss_data, label='Train Loss', color='blue', linewidth=2)
    ax.plot(x, eval_loss_data, label='Eval Loss', color='orange', linewidth=2)

    # 添加标题和轴标签
    ax.set_title('Training and Evaluation Loss Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)

    # 添加网格和坐标轴格式
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)

    # 添加图例并设置位置
    ax.legend(loc='upper right', fontsize=12)

    # 保存图像
    plt.tight_layout()  # 自动调整子图参数，避免标签重叠
    plt.savefig(args.o, format='png', dpi=300)

    # 显示图像
    plt.show()