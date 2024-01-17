import argparse
import os
import numpy as np

"""
python train.py --y coolerCondition & python train.py --y valveCondition & python train.py --y pumpLeak & python train.py --y hydraulicAcc & python train.py --y stableFlag
"""
parser = argparse.ArgumentParser()
parser.add_argument('--y', type=str, default='stableFlag',
                    choices=['coolerCondition', 'valveCondition', 'pumpLeak', 'hydraulicAcc', 'stableFlag'])
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_rate', type=float, default=0.15)
parser.add_argument('--test_rate', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

model_name = args.model_name
result_path = 'result'
model_path = 'models'
os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

out_put = args.y
max_epochs = args.epochs  # 迭代数
val_rate = args.val_rate  # 验证集占比
test_rate = args.test_rate  # 验证集占比
random_state = args.random_state  # 随机数种子
batch_size = args.batch_size  # 数据压缩量
learn_rate = args.lr  # 学习率
step_size = args.step_size  # 学习率递变的步长
gamma = args.gamma  # 学习率递增系数,也即每个epoch学习率变为原来的0.95

model_name += '_' + str(out_put)
npfile = np.load('data/data.npz', allow_pickle=True)
X = npfile['X_normalize']
out_put = 'y_' + out_put
in_Y = npfile[out_put]
