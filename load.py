"""
    Hydraulic_system_status_monitoring-xgboost
    https://blog.csdn.net/Mr_Robert/article/details/84672797
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



# 读取文件
def get_files(dir_path, filename):
    return pd.read_csv(os.path.join(dir_path, filename), sep="\t", header=None)


# 平均周期数据
def mean_conversion(df):
    df = pd.DataFrame(df)
    df1 = df.mean(axis=1)
    return df1

if __name__ == "__main__":
    """
        加载数据
    """
    base_dir = r"data"
    # 加载所有压力传感器数据(sensor data)
    pressureFile1 = get_files(dir_path=base_dir, filename='PS1.txt')
    array = pressureFile1.values[0::, 0::]  # 读取全部行，全部列
    pressureFile2 = get_files(dir_path=base_dir, filename='PS2.txt')
    pressureFile3 = get_files(dir_path=base_dir, filename='PS3.txt')
    pressureFile4 = get_files(dir_path=base_dir, filename='PS4.txt')
    pressureFile5 = get_files(dir_path=base_dir, filename='PS5.txt')
    pressureFile6 = get_files(dir_path=base_dir, filename='PS6.txt')

    # 加载卷流数据(volume flow data)
    volumeFlow1 = get_files(dir_path=base_dir, filename='FS1.txt')
    volumeFlow2 = get_files(dir_path=base_dir, filename='FS2.txt')

    # 加载温度传感器数据(temperature data)
    temperature1 = get_files(dir_path=base_dir, filename='TS1.txt')
    temperature2 = get_files(dir_path=base_dir, filename='TS2.txt')
    temperature3 = get_files(dir_path=base_dir, filename='TS3.txt')
    temperature4 = get_files(dir_path=base_dir, filename='TS4.txt')

    # 加载其余数据：泵效率，振动，冷却效率，冷却功率，效率因数
    pump1 = get_files(dir_path=base_dir, filename='EPS1.txt')
    vibration1 = get_files(dir_path=base_dir, filename='VS1.txt')
    coolingE1 = get_files(dir_path=base_dir, filename='CE.txt')
    coolingP1 = get_files(dir_path=base_dir, filename='CP.txt')
    effFactor1 = get_files(dir_path=base_dir, filename='SE.txt')

    # 从配置文件导入标签数据(profile data)
    profile = get_files(dir_path=base_dir, filename='profile.txt')

    # 将配置文件拆分为相关传感器的标签
    y_coolerCondition = pd.DataFrame(profile.iloc[:, 0])
    y_valveCondition = pd.DataFrame(profile.iloc[:, 1])
    y_pumpLeak = pd.DataFrame(profile.iloc[:, 2])
    y_hydraulicAcc = pd.DataFrame(profile.iloc[:, 3])
    y_stableFlag = pd.DataFrame(profile.iloc[:, 4])

    PS1 = pd.DataFrame(mean_conversion(pressureFile1))
    PS2 = pd.DataFrame(mean_conversion(pressureFile2))
    PS3 = pd.DataFrame(mean_conversion(pressureFile3))
    PS4 = pd.DataFrame(mean_conversion(pressureFile4))
    PS5 = pd.DataFrame(mean_conversion(pressureFile5))
    PS6 = pd.DataFrame(mean_conversion(pressureFile6))

    PS1.columns = ['PS1']
    PS2.columns = ['PS2']
    PS3.columns = ['PS3']
    PS4.columns = ['PS4']
    PS5.columns = ['PS5']
    PS6.columns = ['PS6']

    FS1 = pd.DataFrame(mean_conversion(volumeFlow1))
    FS2 = pd.DataFrame(mean_conversion(volumeFlow2))

    FS1.columns = ['FS1']
    FS2.columns = ['FS2']

    TS1 = pd.DataFrame(mean_conversion(temperature1))
    TS2 = pd.DataFrame(mean_conversion(temperature2))
    TS3 = pd.DataFrame(mean_conversion(temperature3))
    TS4 = pd.DataFrame(mean_conversion(temperature4))

    TS1.columns = ['TS1']
    TS2.columns = ['TS2']
    TS3.columns = ['TS3']
    TS4.columns = ['TS4']

    P1 = pd.DataFrame(mean_conversion(pump1))
    VS1 = pd.DataFrame(mean_conversion(vibration1))
    CE1 = pd.DataFrame(mean_conversion(coolingE1))
    CP1 = pd.DataFrame(mean_conversion(coolingP1))
    SE1 = pd.DataFrame(mean_conversion(effFactor1))

    P1.columns = ['P1']
    VS1.columns = ['VS1']
    CE1.columns = ['CE1']
    CP1.columns = ['CP1']
    SE1.columns = ['SE1']

    # 合并所有dataframes
    X = pd.concat([PS1, PS2, PS3, PS4, PS5, PS6, FS1, FS2, TS1, TS2, TS3, TS4, P1, VS1, CE1, CP1, SE1], axis=1)

    """
        传感器数据可视化
    """
    # 绘制每个传感器的直方图
    X.hist(bins=50, figsize=(20, 15))
    plt.show()

    """
        各种传感器参数之间的相关矩阵
    """
    corr_matrix = X.corr()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))

    # 绘制热图并校正纵横比
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

    """
        数据归一化
    """
    X_normalize = StandardScaler().fit_transform(X)
    np.savez('data/data', X_normalize=X_normalize, y_coolerCondition=y_coolerCondition, y_valveCondition=y_valveCondition,
             y_pumpLeak=y_pumpLeak, y_hydraulicAcc=y_hydraulicAcc, allow_pickle=True)

    print('Processing completed!')
