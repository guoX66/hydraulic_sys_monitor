import json
import os
import platform

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def deal(in_Y):
    y_list = in_Y.tolist()
    state_list = []
    Y = []
    for i in y_list:
        y = i[0]
        if y not in state_list:
            state_list.append(y)
    state_list.sort()
    for i in y_list:
        y = i[0]
        ind = state_list.index(y)
        Y.append([ind])
    Y = np.array(Y)
    return Y, state_list


def bar(i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r训练进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(progress, finsh, need_do, dur), end="")


def write_log(log, txt_list):
    print(log)
    txt_list.append(log + '\r\n')


def write_txt(result_path, txt_list, filename):
    os.makedirs(result_path, exist_ok=True)
    content = ''
    for txt in txt_list:
        content += txt
    with open(f'{result_path}/{filename}.txt', 'w+', encoding='utf8') as f:
        f.write(content)


def process(x, y, batch_size, shuffle):
    seq = []
    for i in range(len(x)):
        train_seq = x[i].unsqueeze(0)

        train_label = y[i][0]
        # train_seq = torch.FloatTensor(train_seq)
        # train_label = torch.FloatTensor(train_label)
        seq.append((train_seq, train_label))

    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return seq


def make_plot(data, mode, filename):
    file_path = 'result/train'
    os.makedirs(file_path, exist_ok=True)
    if mode == 'loss':
        title = 'LOSS'
        path = os.path.join(file_path, 'LOSS-' + filename)
    elif mode == 'acc':
        title = 'ACC'
        path = os.path.join(file_path, 'ACC-' + filename)
    figure(figsize=(12.8, 9.6))
    x = np.arange(1, len(data) + 1)
    plt.plot(x, data, 'red')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title + '-' + filename, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')


def t_model(Dte, txt_list, class_file, model_name):
    gpus = [0, 1]
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    write_log('-' * 43 + '测试结果如下' + '-' * 43, txt_list)
    with open(class_file, 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    right_num = 0
    test_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f'{model_name}')
    model.eval()
    model = model.to(device)  # 将模型迁移到gpu
    try:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    except AssertionError:
        pass
    for data, label in Dte:
        data = data.to(device)
        output = model(data)
        pre = output.argmax(1)
        predict_class = class_dict[int(pre)]
        real_class = class_dict[int(label)]
        if predict_class == real_class:
            right_num += 1
        test_num += 1
    acc = right_num / test_num * 100
    write_log(f'测试总数量为 {test_num} ，错误数量为 {test_num - right_num} ', txt_list)
    write_log(f'总预测正确率为 {acc} %', txt_list)
    return txt_list, acc
