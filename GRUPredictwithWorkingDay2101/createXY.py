"""
首先需要创建用于训练的XY变量，保存到文件夹内
XY变量应当包含24小时步数分布、每天的周数
"""

# 对数据进行reshape
# 结果的shape
# [# 下面是一个训练样本
#  [[0h, 1h, ..., 23h, 0h, ...,23h, ..., ..., 0h, 1h, ..., 23h]     # 21*24列
#   [0 , 0 , ..., 0  , 1 , ...,1  , ..., ..., 6 , 6 , ..., 6  ]     # 21*24列，对应21天的每天星期
#   [0h, 1h, ..., 23h, ..., ..., 0h, ..., 23h]                      # 7*24列
#   [0 , 0 , ..., 0  , ..., ..., 6 , ..., 6  ]],                    # 7*24列，对应7天每天星期
#  # 第二个训练样本
#  [[0h, 1h, ..., 23h, 0h, ...,23h, ..., ..., 0h, 1h, ..., 23h]
#   [0 , 0 , ..., 0  , 1 , ...,1  , ..., ..., 6 , 6 , ..., 6  ]
#   [0h, 1h, ..., 23h, ..., ..., 0h, ..., 23h]
#   [0 , 0 , ..., 0  , ..., ..., 6 , ..., 6  ]], 
#  ...
#  ...
# ]

# 引入必要库函数
import pandas as pd
import numpy as np
import os

# 平移使用到的函数
# 函数输入：data_single_ID, dimension, shift_num
# 函数输出：data_csv_shift
def shiftData(data_single_ID, dimension, shift_num):
    data_multi = pd.DataFrame(columns=np.arange(dimension))
    for i in range(dimension):
        data_multi.iloc[:, dimension - 1 - i] = data_single_ID.shift(shift_num*i)
    data_multi = data_multi.dropna() # 除去NULL，因为序列的起始点是没有历史的
    data_multi_list = np.array(data_multi.values.tolist()).reshape(-1, shift_num, dimension)
    data_csv_shift = []
    for data in data_multi_list:
        data = data.T.tolist()
        data_csv_shift.append(data)
    data_csv_shift = np.array(data_csv_shift)
    return data_csv_shift

# 用于保存npy文件
# 函数输入：filePath, data_multi_list_x, data_multi_list_y
# 函数输出：None
def saveNpy(filePath, data):
    np.save(filePath, data, allow_pickle=True, fix_imports=True)
    print(filePath+" has done!")

# 对所有数据进行合并
# 函数输入：filePath, dimension
# 函数输出：None
def mergedNpy(filePath_read, dimension_1, dimension_2, filePath_save):
    mergeData = []
    files = os.listdir(filePath_read)
    for file in files:
        print(filePath_read+file)
        mergeData.extend(np.load(filePath_read+file).tolist())
        # print(mergeData)
        print(file, "has been merged!")
    # print(mergeData)
    mergeData = np.array(mergeData)
    print(mergeData.shape)
    mergeData = mergeData.reshape(-1, dimension_1, dimension_2)
    np.save(filePath_save, mergeData, allow_pickle=True, fix_imports=True)
    print(filePath_save, "has done!")

if __name__ == "__main__":
    time_num = 24
    start_time = 0
    n_steps_out = 7
    n_steps_in = 21
    file_csv = "../merge_data/merge_resample_interpolation_workingday.csv"
    data_csv = pd.read_csv(file_csv, parse_dates=['date']).set_index('date')
    ID_list = list(data_csv['ID'].unique())

    # 分别处理每个用户
    for id in ID_list:
        
        data_single = data_csv.loc[data_csv["ID"] == id].iloc[:, 2: 27]
        id = str(int(id))
        # data_single:
        # date(index) workingday(0) 0h(1) ...23h(24)
        data_steps_distribution = data_single.iloc[:, 1+start_time:1+start_time+time_num].values
        data_workingday = data_single.iloc[:, 0:1].values.tolist()
        for i in range(len(data_workingday)):
            data_workingday[i] = [data_workingday[i][0] for j in range(time_num)]
        # print(data_steps_distribution)
        data_workingday = np.array(data_workingday)
        # print(data_workingday)
        data_steps_distribution_1d = pd.DataFrame(data_steps_distribution.reshape(-1, 1))
        data_workingday = pd.DataFrame(data_workingday.reshape(-1, 1))

        # data_x
        data_x_steps = shiftData(data_steps_distribution_1d, n_steps_in, time_num)[:-n_steps_out].reshape(-1, n_steps_in*time_num).tolist()
        data_x_workingday = shiftData(data_workingday, n_steps_in, time_num)[:-n_steps_out].reshape(-1, n_steps_in*time_num).tolist()
        data_x = []
        for i in range(len(data_x_steps)):
            data_x.append([data_x_steps[i], data_x_workingday[i]])
        data_x = np.array(data_x)

        # data_y_steps
        data_y_steps = shiftData(data_steps_distribution_1d, n_steps_out, time_num)[n_steps_in:].reshape(-1, n_steps_out*time_num)

        # data_y_workingday
        data_y_workingday = shiftData(data_workingday, n_steps_out, time_num)[n_steps_in:].reshape(-1, n_steps_out*time_num)
        # 保存数据
        file_x_data = "./XY/x_data/"+id+"_x_data.npy"
        file_y_steps = "./XY/y_steps/"+id+"_y_steps.npy"
        file_y_working = "./XY/y_working/"+id+"_y_working.npy"
        saveNpy(file_x_data, data_x)
        saveNpy(file_y_steps, data_y_steps)
        saveNpy(file_y_working, data_y_workingday)

    # 用于存储合并文件
    filePath_read_x_data = "./XY/x_data/"
    filePath_read_y_working = "./XY/y_working/"
    filePath_read_y_steps = "./XY/y_steps/"
    filePath_save_x_data = "./XY/mergeXY/merge_x_data.npy"
    filePath_save_y_working = "./XY/mergeXY/merge_y_working.npy"
    filePath_save_y_steps = "./XY/mergeXY/merge_y_steps.npy"
    mergedNpy(filePath_read_x_data, 2, time_num*n_steps_in, filePath_save_x_data)
    mergedNpy(filePath_read_y_working, 1, time_num*n_steps_out, filePath_save_y_working)
    mergedNpy(filePath_read_y_steps, 1, time_num*n_steps_out, filePath_save_y_steps)