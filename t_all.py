import time
from config import *
from utils import t_model, deal, process, write_txt, write_log

if __name__ == '__main__':
    txt_list = []
    Y, state_list = deal(in_Y)
    x_test = torch.from_numpy(X).float()
    y_test = torch.from_numpy(Y).long()
    Dte = process(x_test, y_test, 1, False)
    st = time.strftime('%Y-%m-%d %H:%M', time.localtime())
    ss_time = time.perf_counter()
    write_log('开始时间: ' + st, txt_list)
    write_log(f'model: {model_name}', txt_list)
    txt_list, acc = t_model(Dte, txt_list, f'result/log/class_id-{out_put}.json', f'{model_path}/{model_name}.pth')
    ed = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    ee_time = time.perf_counter()
    total_time = ee_time - ss_time
    write_log(f"平均每次分类用时 : {round(1000 * total_time / len(Y), 3)} ms", txt_list)
    write_txt(f'{result_path}/test', txt_list, model_name)
