"""
与预测步数相同
使用移位函数，依据前21步预测后七步
相比预测步数，运动模式预测为一向量，需要多进行一步转换

# 1. 根据ID不同对整个数据集进行区分
2. 利用移位函数，产生npy文件，便于使用模型预测
        21 -> 7

"""

# 引入必要库函数
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 平移使用到的函数
# 函数输入：data_single_ID, dimension
# 函数输出：data_csv_shift
def shiftData(data_single_ID, dimension):
    data_multi = pd.DataFrame(columns=np.arange(dimension))
    for i in range(dimension):
        data_multi.iloc[:, dimension - 1 - i] = data_single_ID.shift(24*i)
    data_multi = data_multi.dropna() # 除去NULL，因为序列的起始点是没有历史的
    data_multi_list = np.array(data_multi.values.tolist()).reshape(-1, 24, dimension)
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
    # 每天共有24小时
    time = 24

    # 21步预测7步
    dimension_x = 21
    dimension_y = 7

    # 读取数据
    file_csv = "../merge_data/merge_resample_interpolation.csv"
    data_csv = pd.read_csv(file_csv, parse_dates=['date']).set_index('date').iloc[:, :-1]
    data_csv['ID'] = data_csv['ID'].apply(lambda x: str(int(x)))

    # 获取用户id
    ID_list = list(data_csv['ID'].unique())

    # 分别处理每个用户
    for id in ID_list:
        data = data_csv.loc[data_csv["ID"] == id].iloc[:, 2:]

        # 判断数据是否超过28行
        if data.shape[0] >= 28:
            pass
        else:
            break
        
        # 对数据格式进行处理，便于使用移位函数
        # 将数据转为一维数据，将所有数据首尾连接，且为竖直向量形式
        array_id = data.values.reshape(-1, 1)
        data_df = pd.DataFrame(array_id)

        # 使用移位函数对数据进行移位
        data_multi_list_x = shiftData(data_df, dimension_x)
        data_multi_list_x = data_multi_list_x[:-dimension_y]
        data_multi_list_y = shiftData(data_df, dimension_y)
        data_multi_list_y = data_multi_list_y[dimension_x:]

        # 保存数据
        file_x = "./XY/X/"+id+"_x.npy"
        file_y = "./XY/Y/"+id+"_y.npy"
        saveNpy(file_x, data_multi_list_x)
        saveNpy(file_y, data_multi_list_y)

    # 用于存储合并文件
    x_dimension = 21
    y_dimension = 7
    filePath_read_x = "./XY/X/"
    filePath_read_y = "./XY/Y/"
    filePath_save_x = "./XY/mergeXY/merge_X.npy"
    filePath_save_y = "./XY/mergeXY/merge_Y.npy"
    mergedNpy(filePath_read_x, x_dimension, time, filePath_save_x)
    mergedNpy(filePath_read_y, y_dimension, time, filePath_save_y)