"""
使用gru预测运动分布
输入： 21 * time_num
输出：  7 * time_num

"""

# 导入必要的库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
import os
import glob
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# 需要获取真实步数，真实步数分布，插值归一化数据
# 函数输入：granularity, file_id
# 函数输出：data_steps_28, data_distribution_28, data_interpolation_28, data_label_28, data_label_28
def getData(file_id):
    # 调用文件merge_resample.csv并从中读取id信息
    file_interpolation_csv = "../linear-GRU/merge/merge_resample_interpolation.csv"
    file_resample_csv = "../linear-GRU/merge/merge_resample.csv"

    data_interpolation_csv = pd.read_csv(file_interpolation_csv, parse_dates=['date']).set_index("date")
    data_resample_csv = pd.read_csv(file_resample_csv, parse_dates=['date']).set_index("date")

    data_interpolation_id = data_interpolation_csv.query('ID=='+file_id)
    data_resample_id = data_resample_csv.query('ID=={}'.format(file_id))

    # 调用后28天数据
    data_interpolation_28 = data_interpolation_id.iloc[-28:, 2:26]
    data_resample_28 = data_resample_id.iloc[-28:, 2:26]

    return data_interpolation_28, data_resample_28

# 定义函数用于分割x与y
# 函数输入：data
# 函数输出：data_21, data_7
def splitData(data):
    data_21 = data.iloc[0:21, :]
    data_7 = data.iloc[21:, :]
    return data_21, data_7

# 处理0-23数据
def createInput_0023(data_interpolation_21):
    data_x = data_interpolation_21.values.reshape(1, 21, 24)
    return data_x

# 所有数据进行合并
# 函数输入：file_read_path, file_save_path
# 函数输出：None
def merge(file_read_path, file_save_path):
    all_files = glob.glob(os.path.join(file_read_path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',').set_index('date') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=False)
    df_merged.to_csv(file_save_path)
    print("Merged done!")

if __name__ == "__main__":
    # 时间粒度
    time_num = 24

    # 定义步数
    n_steps_in = 21
    n_steps_out= 7

    # 获取数据用户ID 列表
    data_file = "../linear-GRU/merge/merge_resample_interpolation.csv"
    data_csv = pd.read_csv(data_file).set_index("date")
    # 获取文件ID
    # 找到所有用户ID
    ID = list(data_csv['ID'].unique())
    ID_list = []
    for id in ID:
        id = str(int(id))
        ID_list.append(id)
    ii = 1
    for id in ID_list:
        print(str(ii)+" / "+str(len(ID_list)))
        ii += 1
        data_interpolation, data_resample = getData(id)
        data_interpolation_21, data_interpolation_7 = splitData(data_interpolation)
        data_resample_21, data_resample_7 = splitData(data_resample)
        # print(data_resample_7)

        # 保存步数数据
        data_resample_7.to_csv("./truesteps7/"+id+".csv")

        data_x = createInput_0023(data_interpolation_21)
        x_shape = data_x.shape
        data_x = data_x.reshape(-1, 1)

        # 对数据进行标准化
        std = StandardScaler()
        data_x_std = std.fit_transform(data_x)
        data_x_std = data_x_std.reshape(x_shape)

        # 加载预测模型
        save_path = './GRU_model/linear-GRU.h5'
        GRU_model = load_model(save_path)

        predict_y_std = GRU_model.predict(data_x_std)
        predict_y = std.inverse_transform(predict_y_std)
        predict_y = predict_y.reshape(7, 24)

        # 对有问题的数据（后一时刻步数小于前一时刻的进行处理）
        for i in range(predict_y.shape[0]):
            for j in range(1, predict_y.shape[1]):
                predict_y[i][j] = predict_y[i][j] if predict_y[i][j] >= predict_y[i][j-1] else predict_y[i][j-1]

        predict_y_df = pd.DataFrame(data=predict_y, index=data_resample_7.index, columns=data_resample_7.columns)
        predict_y_df.to_csv("./predictsteps7/"+id+".csv")

    # 合并数据
    file_true_path = "./truesteps7/"
    file_predict_path = "./predictsteps7/"
    merge(file_true_path, "./truesteps.csv")
    merge(file_predict_path, "./predictsteps.csv")  




   