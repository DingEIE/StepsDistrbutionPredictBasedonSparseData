"""
本文件对数据进行统一的预处理
顺序如下：
1. resample
2. continuous
3. interpolation
4. createweek
5. createworking

"""
# scale_val = 1.0
# 导入函数库
import numpy as np
import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# **********首先对数据进行重采样**********
# 将原来的删掉android数据改为处理android数据
# 函数输入：data_csv
# 函数输出：data_csv_after_process  
def processAndroid(data_csv):
    # data_csv = pd.read_csv(filePath+file, parse_dates=['date'])
    # data_csv = data_csv.sort_values(by='date')
    # data_csv.drop_duplicates('date',inplace = True)
    # data_csv = data_csv.set_index("date")
    # del data_csv["Unnamed: 0"]
    # data_csv_ios = data_csv.query('type=="ios"')
    # return data_csv, data_csv_ios
    # data_csv = pd.read_csv(filePath+file, parse_dates=['date'])
    # data_csv = data_csv.sort_values(by='date')
    # data_csv.drop_duplicates('date',inplace = True)
    # data_csv = data_csv.set_index("date")
    # del data_csv["Unnamed: 0"]


    # data_csv_ios = data_csv.query('type=="ios"')
    # data_csv_android = data_csv.query('type=="android"')
    # data_csv_after_process = pd.DataFrame()
    # data_csv_after_process = data_csv_after_process.append(data_csv_ios)
    # # 下面处理安卓数据
    # # 将安卓采样数据统一减去一天采样的最小值，如果最大值的column比最小值的column小，该行数据删除
    # data_csv_android['max_idx'] = data_csv_android.iloc[:, 2:].idxmax(axis=1) #求一行的最大值对应的索引
    # data_csv_android['min_idx'] = data_csv_android.iloc[:, 2:-1].idxmin(axis=1) #求一行的最大值对应的索引
    # data_csv_android['min_val']= data_csv_android.iloc[:, 2:-2].min(axis=1) #取出最小值
    # for i in range(data_csv_android.shape[0]):
    #     if int(data_csv_android.iloc[i:i+1, -2].values[0]) < int(data_csv_android.iloc[i:i+1, -3].values[0]):
    #         data_csv_android.iloc[i:i+1, 1:-3] = data_csv_android.iloc[i:i+1, 1:-3].apply(lambda x: x-data_csv_android.iloc[i,-1])
    #         data_csv_after_process = data_csv_after_process.append(data_csv_android.iloc[i:i+1, 0:-3])
    # return data_csv_after_process
    data_csv['min_val'] = data_csv.iloc[:, 2:].min(axis=1)
    data_csv.iloc[:, 1:2] = data_csv.iloc[:, 1:2].apply(lambda x: x-data_csv.iloc[:,-1])
    data_csv.iloc[:, 2:3] = data_csv['min_val']
    del data_csv['min_val']
    miss_num = pd.DataFrame(1442 - data_csv.count(axis=1))
    data_csv["sparsity"] = miss_num.values/1440
    data_csv.iloc[:, 2:-1] = data_csv.iloc[:, 2:-1].interpolate(axis=1)
    # print(data_csv)
    data_csv.iloc[:, 2:-1] = data_csv.iloc[:, 2:-1].diff(periods=1, axis=1)
    # data_csv = data_csv.fillna(0)
    data_csv.iloc[:, 2] = 0
    
    data_csv[data_csv._get_numeric_data() < 0] = np.nan
    data_csv = data_csv.dropna(axis=0)
    # data_csv[data_csv._get_numeric_data() == 0] = np.nan
    
    data_csv.iloc[:, 2:-1] = data_csv.iloc[:, 2:-1].cumsum(axis=1)
    return data_csv

    # data_csv_ios = data_csv.query('type=="ios"')
    # data_csv_android = data_csv.query('type=="android"')
    # data_csv_after_process = pd.DataFrame()
    # data_csv_after_process = data_csv_after_process.append(data_csv_ios)
    # # 下面处理安卓数据
    # # 将安卓采样数据统一减去一天采样的最小值，如果最大值的column比最小值的column小，该行数据删除
    # data_csv_android['max_idx'] = data_csv_android.iloc[:, 2:].idxmax(axis=1) #求一行的最大值对应的索引
    # data_csv_android['min_idx'] = data_csv_android.iloc[:, 2:-1].idxmin(axis=1) #求一行的最大值对应的索引
    # data_csv_android['min_val']= data_csv_android.iloc[:, 2:-2].min(axis=1) #取出最小值
    # for i in range(data_csv_android.shape[0]):
    #     if int(data_csv_android.iloc[i:i+1, -2].values[0]) < int(data_csv_android.iloc[i:i+1, -3].values[0]):
    #         data_csv_android.iloc[i:i+1, 1:-3] = data_csv_android.iloc[i:i+1, 1:-3].apply(lambda x: x-data_csv_android.iloc[i,-1])
    #         data_csv_after_process = data_csv_after_process.append(data_csv_android.iloc[i:i+1, 0:-3])
    # return data_csv_after_process

# 判断数据能否被保留
# 天数多于28天
# 不含Android用户
# 不含空缺天数
# 定义函数，判断数据能否被保留
# 函数输入：data_csv, data_csv_ios
# 函数输出：data_csv
def chooseDate(data_csv):
    noneDF = pd.DataFrame()
    if data_csv.shape[0] >= 28:
        return data_csv
    else:
        return noneDF

# 对数据进行重采样
# 原1440个数据，每60个数据进行一次采样，获得最终24个数据
# 函数输入:data_csv
# 函数输出:data_csv_resample
def resample24(data_csv):
    data_csv_resample = pd.DataFrame()
    data_csv_resample.index = data_csv.index
    data_csv_resample["steps"] = data_csv["steps"]
    
    # print(data_csv)
    for i in range(0, 24):
        data = data_csv.iloc[:, 60*i+2:60*i+62].max(axis=1)
        data_csv_resample[str(i)] = data.values
    data_csv_resample["0"] = data_csv_resample["0"].fillna(0)
    data_csv_resample["23"] = data_csv_resample["steps"].values
    data_csv_resample['sparsity'] = data_csv["sparsity"]
    data_csv_resample[[str(i) for i in range(24)]] = data_csv_resample[[str(i) for i in range(24)]].astype(int)
    return data_csv_resample



def resample():
    # 首先读取文件
    # 要处理的文件存储路径为"././detailed_steps/"
    filePath = "./detailed_steps/"
    files = os.listdir(filePath)
    for file in files:
        print(files.index(file)+1, '/', len(files))
        data_csv = pd.read_csv(filePath+file, parse_dates=['date'])
        data_csv = data_csv.sort_values(by='date')
        data_csv.drop_duplicates('date',inplace = True)
        data_csv = data_csv.set_index("date")
        del data_csv["Unnamed: 0"]
        data_csv = chooseDate(data_csv)# 先选择数据
        if data_csv.shape[0]:
            data_csv_after_processAndroid = processAndroid(data_csv)
            data_csv_after_processAndroid = chooseDate(data_csv_after_processAndroid)
            # 这里添加对数据进行处理
            # 这里继续选择数据
            # print(data_csv.shape[0])
            if data_csv_after_processAndroid.shape[0]:
                data_csv_resample = resample24(data_csv_after_processAndroid)

                # 将用户ID插入数据中
                ID = file.replace(".csv", "")
                data_csv_resample.insert(0, "ID", ID)

                # 将数据输出,写入文件
                file_Path = "./DataAfterClean/resample/resample_"+file
                data_csv_resample.to_csv(file_Path)
                print(file+" has been saved!")
            else:
                print(file+" Pass1")
        else:
            print(file+" Pass2")



# **********采集数据中超过连续28天的数据**********
# 此文件用于提取数据中连续28天及以上的数据
import os
import pandas as pd
import numpy as np
import glob
import re


def getData(file_read, file_save):
    df = pd.read_csv(file_read, parse_dates=['date'])
    time_diff = df['date'].diff(1).values.tolist()
    time_error = [0]
    for i in time_diff:
        # print(i)
        if i != 86400000000000 and i != None:
            # print(i)
            time_error.append(time_diff.index(i, time_error[-1]+1))
    time_error.append(len(time_diff)-1)
    # print(time_error)
    ii = 0
    for i in range(1, len(time_error)):
        if time_error[i]-time_error[i-1] >= 28:
            df_ii = df.iloc[time_error[i-1]:time_error[i], :]
            ID = str(df_ii.iloc[0, 1])+"0"+str(ii)
            df_ii[['ID']] = ID
            df_ii.to_csv(file_save.replace('.csv', '_'+ID+'.csv'), index=False)
            ii += 1
            print(file_save.replace('.csv', '_'+ID+'.csv')+" has been saved! ")

# 所有数据进行合并
# 函数输入：file_read_path, file_save_path
# 函数输出：None
def merge(file_read_path, file_save_path):
    all_files = glob.glob(os.path.join(file_read_path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',').set_index('date') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=False)
    df_merged.to_csv(file_save_path)
    print("Merged done!")

def continuous():
    file_read_path = "./DataAfterClean/resample/"
    files = os.listdir(file_read_path)
    for file in files:
        file_read = file_read_path+file
        file_save = "./DataAfterClean/continuous28/"+file
        getData(file_read, file_save)
    # 将所有数据写入一个文件,方便操作
    file_read_path = "./DataAfterClean/continuous28/"
    file_save_path = "./merge_data/merge_resample_.csv"
    merge(file_read_path, file_save_path)


# *********选择数据中方差小于一定数值的部分*********
# 将所有数据文件的总步数方差输出，选择最小的60%作为易于预测的部分对数据进行预测
def getEasyPredictData(scale_val):
    files = os.listdir("./DataAfterClean/continuous28/")
    std_steps_map = {}
    std_steps_list = []
    good_ID_list = []
    bad_ID_list = []
    for file in files:
        df = pd.read_csv("./DataAfterClean/continuous28/"+file).set_index("date")
        steps_df = df['steps'].values
        std_steps = np.std(steps_df)
        # print(std_steps)
        # print(file)
        id = re.search("_[0-9]*.csv", file).group(0).replace('_', '').replace('.csv', '')
        # print(id)
        std_steps_map[id] = std_steps
        std_steps_list.append(std_steps)
    # print(std_steps_map)
    std_steps_list.sort()
    length_list = len(std_steps_list)
    compare_val = std_steps_list[int(length_list*scale_val)-1]
    for key, value in std_steps_map.items():
        if value <= compare_val:
            good_ID_list.append(key)
        else:
            bad_ID_list.append(key)
    print(len(std_steps_list))
    print("The number of values that are easily predictable:", len(good_ID_list))
    print("The number of values that are hard predictable:", len(bad_ID_list))
    file_csv_ = "./merge_data/merge_resample_.csv"
    file_csv = "./merge_data/merge_resample.csv"
    df_csv = pd.read_csv(file_csv_)
    df_output = pd.DataFrame()
    ID_list = list(df_csv['ID'].unique())
    df_csv.set_index('date')
    for id in ID_list:
        if str(id) in good_ID_list:
            df_output = df_output.append(df_csv.query('ID=={}'.format(id)))
    df_output.to_csv(file_csv, index=False)

# **********对稀缺数据进行线性插值**********
# 导入函数库
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'

#hist函数可以直接绘制直方图
#参数有四个，第一个必选
#arr: 需要计算直方图的一维数组
#bins: 直方图的柱数，可选项，默认为10
#facecolor: 直方图颜色
#alpha: 透明度
#返回值为n: 直方图向量，是否归一化由参数设定；bins: 返回各个bin的区间范围；patches: 返回每个bin里面包含的数据，是一个list
def plotSparsity(sparsity):
    sparsity_list = []
    for value in sparsity:
        value = round(value*100)
        sparsity_list.append(value)
    plt.figure(figsize=(9, 6))
    n, bins, patches = plt.hist(sparsity_list, bins=256, weights= [1./ len(sparsity_list)] * len(sparsity_list), facecolor='blue', alpha=0.75)  
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('Sparsity')
    plt.ylabel('Ratio')
    plt.title("Sparse Data Distribution")
    plt.show()

# 对缺失步数数据进行线性插补
# 函数输入:data_csv
# 函数输出:data_without_miss
def interpolateSteps(data_csv):
    data_without_miss = data_csv
    data_without_miss.iloc[:, 2:26] = data_csv.iloc[:, 2:26].interpolate(axis=1)
    # data_without_miss = data_without_miss.dropna()
    data_without_miss[[str(i) for i in range(24)]] = data_without_miss[[str(i) for i in range(24)]].astype(int)
    return data_without_miss

# 绘制步数分布图
# 函数输入：data_without_miss
# 函数输出: data_steps
def stepsPlot(data_without_miss):
    steps_list = []
    steps = data_without_miss.iloc[:, 2:26]
    for i in range(24):
        temp = steps[str(i)].mean()
        steps_list.append(temp)
    x = [i for i in range(24)]
    data_steps = pd.DataFrame()
    data_steps.insert(0, "Time", x)
    data_steps = data_steps.set_index("Time")
    data_steps.insert(0, "steps", steps_list)
    plt.figure(figsize=(18, 12))
    plt.plot(data_steps["steps"])
    plt.xticks(np.arange(0, 25, 1))
    plt.xlabel('Time(h)')
    plt.ylabel('Steps')
    plt.show()
    return data_steps

# 绘制步数增长分布图
# 函数输入:data_steps
# 函数输出:None
def plotStepsDiff(data_steps):
    data_steps_diff = data_steps.diff().iloc[1:23]
    max_steps = data_steps["steps"].max()
    data_steps_diff_radio = data_steps_diff.apply(lambda x: x/max_steps)
    plt.figure(figsize=(9, 6))
    plt.plot(data_steps_diff_radio["steps"])
    plt.xticks(np.arange(0, 25, 1))
    plt.xlabel('Time(h)')
    plt.ylabel('Ratio')
    plt.title("Step Increase Display")
    plt.show()

# 输出归一化的数据
# 函数输入：data_csv
# 函数输出：data_csv
def ScalerSteps(data_csv):
    for i in range(24):
        steps = data_csv["23"]
        data_csv[str(i)] = pd.DataFrame(data_csv[str(i)]).apply(lambda x: x / steps)
        data_csv[str(i)] = data_csv[str(i)].round(3)
    return data_csv

def interpolation():
    # 对采样得到的数据进行插补，使用线性插补方式去除空缺值
    filePath = "./merge_data/merge_resample.csv"
    data_csv = pd.read_csv(filePath).set_index("date")
    sparsity = data_csv["sparsity"].values
    plotSparsity(sparsity)
    data_without_miss = interpolateSteps(data_csv)
    data_steps = stepsPlot(data_without_miss)
    plotStepsDiff(data_steps)
    file_save_path = "./merge_data/merge_resample_interpolation.csv"
    data_without_miss.to_csv(file_save_path)
    print("Done")


# **********创建星期数据**********
import pandas as pd


# 函数输入：data_csv
# 函数输出: data_csv
def CreateWeek(data_csv):
    data_date = data_csv.index.dayofweek

    data_csv.insert(2, 'weekday', data_date)
    # data_week = pd.get_dummies(data_csv['weekday'])
    # del data_csv['weekday']
    # data_csv.insert(26, 'Mon', data_week['Monday'])
    # data_csv.insert(27, 'Tue', data_week['Tuesday'])
    # data_csv.insert(28, 'Wed', data_week['Wednesday'])
    # data_csv.insert(29, 'Thu', data_week['Thursday'])
    # data_csv.insert(30, 'Fri', data_week['Friday'])
    # data_csv.insert(31, 'Sat', data_week['Saturday'])
    # data_csv.insert(32, 'Sun', data_week['Sunday'])
    
    return data_csv

def createweek():
    file = "./merge_data/merge_resample_interpolation.csv"
    data_csv = pd.read_csv(file, parse_dates=['date']).set_index('date')
    data_csv = CreateWeek(data_csv)
    # 此时的data_csv
    # date(index) ID(0) steps(1) 0 1 ... 23(25) Mon(26) ... Sun(32) sparsity(33)
    data_csv.to_csv("./merge_data/merge_resample_interpolation_week.csv")
    print("DONE")


# **********对数据创造工作日数据**********
import pandas as pd

# 函数输入：data_csv
# 函数输出: data_csv
def CreateWorkingDay(data_csv):
    data_date = data_csv.index.dayofweek
    data_date = pd.DataFrame(data_date)
    data_date.iloc[:, 0] = data_date.iloc[:, 0].apply(lambda x: int(x/5))
    data_csv.insert(2, 'workingday', data_date)
    # data_week = pd.get_dummies(data_csv['weekday'])
    # del data_csv['weekday']
    # data_csv.insert(26, 'Mon', data_week['Monday'])
    # data_csv.insert(27, 'Tue', data_week['Tuesday'])
    # data_csv.insert(28, 'Wed', data_week['Wednesday'])
    # data_csv.insert(29, 'Thu', data_week['Thursday'])
    # data_csv.insert(30, 'Fri', data_week['Friday'])
    # data_csv.insert(31, 'Sat', data_week['Saturday'])
    # data_csv.insert(32, 'Sun', data_week['Sunday'])
    
    return data_csv

def createworkingday():
    file = "./merge_data/merge_resample_interpolation.csv"
    data_csv = pd.read_csv(file, parse_dates=['date']).set_index('date')
    data_csv = CreateWorkingDay(data_csv)
    # 此时的data_csv
    # date(index) ID(0) steps(1) 0 1 ... 23(25) Mon(26) ... Sun(32) sparsity(33)
    data_csv.to_csv("./merge_data/merge_resample_interpolation_workingday.csv")
    print("DONE")



# *********对以上的函数进行汇总**********
def dataprepare(scale_val):
    # print("*"*10+"RESAMPLE"+"*"*10)
    # resample()
    # print("*"*10+"CONTINUOUS"+"*"*10)
    # continuous()
    print("*"*10+"GETEASYPREDICT"+"*"*10)
    getEasyPredictData(scale_val)
    print("*"*10+"INTERPOLATION"+"*"*10)
    interpolation()
    print("*"*10+"CREATEWEEK"+"*"*10)
    createweek()
    print("*"*10+"CREATEWORKINGDAY"+"*"*10)
    createworkingday()




if __name__ == "__main__":
    scale_val = 1.0
    dataprepare(scale_val)