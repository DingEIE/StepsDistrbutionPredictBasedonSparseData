"""
创建可用于训练的data
"""

import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler

time_num = 24
start_time = 0
n_steps_out = 7
n_steps_in = 21
# scale_val = 1.0
from LSHP import scale_val

# 用于创建不存在的文件夹
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
	else:
		pass

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

# 用于保存npy文件
# 函数输入：filePath, data_multi_list_x, data_multi_list_y
# 函数输出：None
def saveNpy(filePath, data):
    np.save(filePath, data, allow_pickle=True, fix_imports=True)
    print(filePath+" has done!")

def createXY():
    
    file_csv = "./merge_data/merge_resample_interpolation_week.csv"
    data_csv = pd.read_csv(file_csv, parse_dates=['date']).set_index('date')
    ID_list = list(data_csv['ID'].unique())

    # 分别处理每个用户
    for id in ID_list:
        
        data_single = data_csv.loc[data_csv["ID"] == id].iloc[:, 2: 27]
        id = str(int(id))
        # data_single:
        # date(index) weekday(0) 0h(1) ...23h(24)
        data_steps_distribution = data_single.iloc[:, 1+start_time:1+start_time+time_num].values

        # 创建星期数据
        data_weekday = data_single.iloc[:, 0:1].values.tolist()
        for i in range(len(data_weekday)):
            data_weekday[i] = [data_weekday[i][0] for j in range(time_num)]
        # print(data_steps_distribution)
        data_weekday = np.array(data_weekday)

        # 创建时间数据
        data_hour = [i for i in range(time_num)]*n_steps_in
        data_hour = np.array(data_hour)


        # print(data_weekday)
        data_steps_distribution_1d = pd.DataFrame(data_steps_distribution.reshape(-1, 1))
        data_weekday = pd.DataFrame(data_weekday.reshape(-1, 1))
        data_weekday.iloc[:, 0] = data_weekday.iloc[:, 0].apply(lambda x: math.cos(2*math.pi/7*x))

        # data_x
        data_x_steps = shiftData(data_steps_distribution_1d, n_steps_in, time_num)[:-n_steps_out].reshape(-1, n_steps_in*time_num).tolist()
        data_x_weekday = shiftData(data_weekday, n_steps_in, time_num)[:-n_steps_out].reshape(-1, n_steps_in*time_num).tolist()
        data_x = []
        for i in range(len(data_x_steps)):
            data_x.append([data_x_steps[i], data_x_weekday[i], data_hour])
        data_x = np.array(data_x)

        # data_y_steps
        data_y_steps = shiftData(data_steps_distribution_1d, n_steps_out, time_num)[n_steps_in:].reshape(-1, n_steps_out*time_num)

        # data_y_weekday
        data_y_weekday = shiftData(data_weekday, n_steps_out, time_num)[n_steps_in:].reshape(-1, n_steps_out*time_num)
        # 保存数据
        mkdir("./XY/x_data/")
        mkdir("./XY/y_steps/")
        mkdir("./XY/y_week/")
        file_x_data = "./XY/x_data/"+id+"_x_data.npy"
        file_y_steps = "./XY/y_steps/"+id+"_y_steps.npy"
        file_y_week = "./XY/y_week/"+id+"_y_week.npy"
        saveNpy(file_x_data, data_x)
        saveNpy(file_y_steps, data_y_steps)
        saveNpy(file_y_week, data_y_weekday)

    # 用于存储合并文件
    filePath_read_x_data = "./XY/x_data/"
    filePath_read_y_week = "./XY/y_week/"
    filePath_read_y_steps = "./XY/y_steps/"
    mkdir("./XY/mergeXY/")
    filePath_save_x_data = "./XY/mergeXY/merge_x_data.npy"
    filePath_save_y_week = "./XY/mergeXY/merge_y_week.npy"
    filePath_save_y_steps = "./XY/mergeXY/merge_y_steps.npy"
    mergedNpy(filePath_read_x_data, 3, time_num*n_steps_in, filePath_save_x_data)
    mergedNpy(filePath_read_y_week, 1, time_num*n_steps_out, filePath_save_y_week)
    mergedNpy(filePath_read_y_steps, 1, time_num*n_steps_out, filePath_save_y_steps)

class DataUtils:
    def __init__(self) -> None:
        if os.path.exists("./XY/mergeXY"+str(scale_val)+"/merge_x_data.npy"):
            pass
        else:
            createXY()

        # dataX结构：[[步数], [星期(cos形式)], [小时]] 3X504
        self.dataXraw = np.load("./XY/mergeXY"+str(scale_val)+"/merge_x_data.npy")
        self.dataYstepsraw = np.load("./XY/mergeXY"+str(scale_val)+"/merge_y_steps.npy").reshape(-1, n_steps_out*time_num)
        self.dataYweekraw = np.load("./XY/mergeXY"+str(scale_val)+"/merge_y_week.npy").reshape(-1, n_steps_out*time_num)

        self.dataXsteps = self.dataXraw[:, 0, :]
        self.dataXweek = self.dataXraw[:, 1, :]
        self.dataXhour = self.dataXraw[:, 2, :]

        self.dataXstepsstd = StandardScaler()
        dataXweekstd = StandardScaler()
        dataXhourstd = StandardScaler()
        self.dataYstepsstd = StandardScaler()
        dataYweekstd = StandardScaler()
        
        self.dataXsteps_std = self.dataXstepsstd.fit_transform(self.dataXsteps)
        self.dataXweek_std = dataXweekstd.fit_transform(self.dataXweek.T).T
        self.dataXhour_std = dataXhourstd.fit_transform(self.dataXhour.T).T

        self.dataYsteps_std = self.dataYstepsstd.fit_transform(self.dataYstepsraw)
        self.dataYweek_std = dataYweekstd.fit_transform(self.dataYweekraw.T).T

        Yhour = np.array([[i for i in range(24)]])
        dataYhourstd = StandardScaler()
        self.dataYhour_std = dataYhourstd.fit_transform(Yhour.T).T

        """
        >>> data.dataXraw.shape
        (9762, 3, 504)
        >>> data.dataYstepsraw.shape
        (9762, 168)
        >>> data.dataYweekraw.shape
        (9762, 168)
        >>> data.dataXsteps_std.shape
        (9762, 504)
        >>> data.dataXweek_std.shape
        (9762, 504)
        >>> data.dataXhour_std.shape
        (9762, 504)
        >>> data.dataYsteps_std.shape
        (9762, 168)
        >>> data.dataYweek_std.shape
        (9762, 168)
        >>> data.dataYhour_std.shape
        (1, 24)
        """

        self.dataXsteps_std = self.dataXsteps_std.reshape(-1, n_steps_in, 1, time_num)
        self.dataXweek_std = self.dataXweek_std.reshape(-1, n_steps_in, 1, time_num)
        self.dataXhour_std = self.dataXhour_std.reshape(-1, n_steps_in, 1, time_num)
        self.dataYsteps_std = self.dataYsteps_std.reshape(-1, n_steps_out, time_num)
        self.dataYweek_std = self.dataYweek_std.reshape(-1, n_steps_out, time_num)


        """
        >>> data.dataXsteps_std.shape
        (9762, 21, 1, 24)
        >>> data.dataXweek_std.shape
        (9762, 21, 1, 24)
        >>> data.dataXhour_std.shape
        (9762, 21, 1, 24)
        >>> data.dataYsteps_std.shape
        (9762, 7, 24)
        >>> data.dataYweek_std.shape
        (9762, 7, 24)
        """
        
        self.dataX = np.concatenate((self.dataXsteps_std, self.dataXweek_std, self.dataXhour_std), axis=2)

        """
        >>> data.dataX.shape
        (9762, 21, 3, 24)
        """

        self.label = self.dataYstepsraw.reshape(-1, n_steps_out, time_num)[:, 0, :].reshape(-1, 1, time_num)
        self.labelstd = StandardScaler()
        self.label_std = self.labelstd.fit_transform(self.label.reshape(-1, time_num)).reshape(-1, 1, time_num)


def main():
    data = DataUtils()
    print(data.dataX.shape)
    print(data.dataX)

if __name__ == "__main__":
    main()




