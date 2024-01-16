import json
import time
from torch import nn
from torch.optim.lr_scheduler import StepLR
from config import *
from model import ResNet
from utils import process, bar, make_plot, deal, write_log, write_txt, t_model
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    Y, state_list = deal(in_Y)
    x_train, x_test = train_test_split(X, test_size=val_rate + test_rate,
                                       random_state=random_state)
    y_train, y_test = train_test_split(Y, test_size=val_rate + test_rate,
                                       random_state=random_state)

    x_val, x_test = train_test_split(x_test, test_size=int(test_rate * len(X)),
                                     random_state=random_state)
    y_val, y_test = train_test_split(y_test, test_size=int(test_rate * len(X)),
                                     random_state=random_state)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    Dtr = process(x_train, y_train, batch_size, True)
    Dva = process(x_val, y_val, batch_size, True)
    Dte = process(x_test, y_test, 1, False)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速运算
    net = ResNet(in_channels=1, classes=len(state_list))
    model = net.to(device)  # 将模型迁移到gpu
    best_acc = 0  # 起步正确率
    ba_epoch = 1  # 标记最高正确率的迭代数
    min_lost = np.Inf  # 起步loss
    ml_epoch = 1  # 标记最低损失的迭代数
    acc_list = []
    loss_list = []
    txt_list = []
    st = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    ss_time = time.perf_counter()
    write_log('开始时间: ' + st, txt_list)
    write_log(f'model : {model_name}', txt_list)
    write_log(f'epochs : {max_epochs}', txt_list)
    write_log(f'batch_size : {batch_size}', txt_list)
    write_log(f'learn_rate : {learn_rate}', txt_list)
    write_log(f'step_size : {step_size}', txt_list)
    write_log(f'gamma : {gamma}', txt_list)
    write_log(f'random_state : {random_state}', txt_list)
    write_log(f'device  : {device}', txt_list)

    loss_fn = nn.CrossEntropyLoss().to(device)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 可调超参数
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    try:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    except AssertionError:
        pass
    for i in range(max_epochs):
        print(f"--------第{i + 1}轮训练开始---------")
        model.train()
        start_time = time.perf_counter()
        for j, [imgs, targets] in enumerate(Dtr):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化
            bar(j + 1, len(Dtr), start=start_time)
        print()
        scheduler.step()
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            n_total_val = 0
            for j, data in enumerate(Dva):
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                n_total_val += len(targets)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
                print("\r----------验证集验证中({:^3.2f}%)-----------".format((j + 1) * 100 / len(Dva)),
                      end="")
            print()
            acc = float(total_accuracy * 100 / n_total_val)
            print("验证集上的loss：{:.6f}".format(total_val_loss))
            print("验证集上的正确率：{:.6f}%".format(acc))
            acc_list.append(round(acc, 3))
            loss_list.append(round(total_val_loss, 3))

            if acc > best_acc:  # 保存迭代次数中最好的模型
                print("根据正确率，已修改模型")
                best_acc = acc
                ba_epoch = i + 1
                torch.save(model, f'{model_path}/{model_name}.pth')
            if total_val_loss < min_lost:
                print("根据损失，已修改模型")
                ml_epoch = i + 1
                min_lost = total_val_loss
                torch.save(model, f'{model_path}/{model_name}.pth')

    class_id = {i: state_list[i] for i in range(len(state_list))}
    json_str = json.dumps(class_id)
    with open(f'result/log/class_id-{out_put}.json', 'w') as json_file:
        json_file.write(json_str)
    write_log(f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(acc, 6)}%', txt_list)
    write_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(min_lost, 6)}', txt_list)
    ed = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    ee_time = time.time()
    total_time = ee_time - ss_time
    write_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)

    txt_list, acc = t_model(Dte, txt_list, f'result/log/class_id-{out_put}.json', f'{model_path}/{model_name}.pth')
    write_txt(f'{result_path}/log', txt_list, model_name)
    make_plot(loss_list, 'loss', model_name)
    make_plot(acc_list, 'acc', model_name)


