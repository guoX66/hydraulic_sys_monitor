# 液压系统状态监控与分类

根据压力、体积流量和温度等过程值，得出四个液压元件（冷却器、阀门、泵和蓄能器）的状况。



## 环境部署

首先需安装 python>=3.10.2，然后安装torch>=2.1.1,torchaudio>=2.1.1 torchvision>=0.16.1

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

安装后可使用以下命令依次查看torch，cuda版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda)"
```

安装其他环境依赖

```
pip install -r requirements.txt
```



## 数据处理

将数据保存至data文件夹中，来源：[液压系统的状态监测 - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

```
--hydraulic_sys_monitor
    --data
        --PS1.txt
        --PS2.txt
        ...
```

运行数据加载与处理程序，处理后的数据将保存在data/data.npz中，参考[利用xgboost算法对液压系统的状态进行预测并分析影响因素重要性_液压油缸工作状态预测-CSDN博客](https://blog.csdn.net/Mr_Robert/article/details/84672797)

```bash
python load.py
```

data文件夹已经带有处理好的data.npz



## 模型训练

使用一维Resnet50网络，有其他网络也可以加入model.py中使用

监控变量可选择coolerCondition、valveCondition、pumpLeak、hydraulicAcc、stableFlag，训练得到的模型保存在models文件夹中

```bash
python train.py --y coolerCondition --epochs 50 --batch_size 32 --val_rate 0.15 --test_rate 0.05 --lr 0.001 --step_size 1 --gamma 0.95 --random_state 42
```

或者依次训练四种变量（使用）

```bash
python train.py --y coolerCondition & python train.py --y valveCondition & python train.py --y pumpLeak & python train.py --y hydraulicAcc & python train.py --y stableFlag
```

训练过程保存在result/train中

训练结果保存在result/log中



## 模型测试

在所有数据上进行测试，计算正确率

监控变量可选择coolerCondition、valveCondition、pumpLeak、hydraulicAcc

```bash
python t_all.py --y coolerCondition
```

或

```bash
python t_all.py --y coolerCondition & python t_all.py --y valveCondition & python t_all.py --y pumpLeak & python t_all.py --y hydraulicAcc & python t_all.py --y stableFlag
```

测试结果保存在result/test中
