"""
使用gru预测运动分布
输入： 21 * time_num
输出：  7 * time_num

滚动预测

"""

# 导入必要的库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, GRU, concatenate, BatchNormalization, Dropout
import os
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import r2_score
# from tensorflow.keras.models import load_weights
import warnings
warnings.filterwarnings('ignore')

# 时间粒度
time_num = 24

# 定义步数
n_steps_in = 21
n_steps_out= 7

units = 1000
epochs = 1000
# 定义模型存储位置
weight_save_path = './model/GRU_1000_4-1000-roll-adam.h5'
# 计算smape
def smape(data_pred, data_true):
    data_pred = np.array(data_pred)
    data_true = np.array(data_true)
    smape_val = np.mean(np.abs(2*(data_pred - data_true) / (data_true+1+data_pred)))
    return smape_val

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

import math
def calc_corr(a,b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
    corr_factor = cov_ab/sq
    return corr_factor

# 定义函数用于GRU模型训练
# 函数输入： std_x_steps_train, std_x_working_train, std_y_working_train, std_y_steps_train
# 函数输出：save_path
def trainModel(std_x_steps_train, std_x_working_train, std_y_working_train, std_y_steps_train):
    # 配置训练模型
    # 使用GRU模型

    # # 首先将std_x_steps_train与std_x_working_train相连
    # x_data = concatenate([std_x_steps_train, std_x_working_train], axis=1)
    __input1_ = Input(shape=(2, n_steps_in*time_num))
    _input2_ = Input(shape=(1, n_steps_out*time_num))
    _input1_ = __input1_

    # 要对input2_数据进行截断，每次训练取24位
    # 每次训练结束输出24位数据要和input1_进行结合，去除input1_开始的24位
    # 写一个for循环，每次input2_都取[24*i:24*i+24]的数据
    # 在for循环内，预测未来一天24h的步数分布
    # 预测结果要append在一起作为输出
    # 预测结果要和input2_[24*i:24*i+24]结合
    # 将结合后的数据输入input1_中，然后取input1_[24:]的数据作为输入进行下一次循环
    for i in range(n_steps_out):
        input2_ = _input2_[:, 0:1, 24*i:24*i+24]
        # print(input2_.get_shape().as_list())
        input1_ = Dropout(0.2, input_shape=(2, n_steps_in*time_num))(__input1_)
        hidden_state=GRU(units=units,                                                 # RNN层中的单元数
                        activation='relu',                                            # 激活函数
                        recurrent_activation='sigmoid',                               # 循环步骤中的激活函数
                        return_sequences=True,                                       # 是否返回最后的输出
                        input_shape=(2, n_steps_in*time_num))(input1_)                # 输入x的形状
        std_hidden_state = BatchNormalization(axis=1)(hidden_state)
        input1_ = Dropout(0.2)(std_hidden_state)
        hidden_state=GRU(units=units,                                                 # RNN层中的单元数
                        activation='relu',                                            # 激活函数
                        recurrent_activation='sigmoid',                               # 循环步骤中的激活函数
                        return_sequences=False)(input1_)                                            # 是否返回最后的输出
        std_hidden_state = BatchNormalization(axis=1)(hidden_state)
        hidden_state_dropout = Dropout(0.2)(std_hidden_state)
        hidden_state_dense = Dense(1*time_num)(hidden_state_dropout)
        hidden_state_reshape = tf.reshape(hidden_state_dense, [-1, 1, 1*time_num])
        std_hidden_state_reshape = BatchNormalization(axis=2, input_shape=(1, 1*time_num))(hidden_state_reshape)
        input2 = concatenate([std_hidden_state_reshape, input2_], axis=1)

        output_GRU_input = Dropout(0.2, input_shape=(2, 1*time_num))(input2)
        output_GRU = GRU(units=units,                                                 # RNN层中的单元数
                        activation='relu',                                            # 激活函数
                        recurrent_activation='sigmoid',                               # 循环步骤中的激活函数
                        return_sequences=True,                                       # 是否返回最后的输出
                        input_shape=(2, 1*time_num))(output_GRU_input)                # 输入x的形状
        std_output_GRU = BatchNormalization(axis=1)(output_GRU)
        output_GRU_input = Dropout(0.2)(std_output_GRU)
        output_GRU = GRU(units=units,                                                 # RNN层中的单元数
                        activation='relu',                                            # 激活函数
                        recurrent_activation='sigmoid',                               # 循环步骤中的激活函数
                        return_sequences=False)(output_GRU_input)                                       # 是否返回最后的输出
        std_output_GRU = BatchNormalization(axis=1)(output_GRU)
        output_GRU_dropout = Dropout(0.2)(std_output_GRU)
        output_GRU_dense = Dense(1*time_num)(output_GRU_dropout)
        output_GRU_dense_reshape = tf.reshape(output_GRU_dense, [-1, 1, 1*time_num])
        std_output_GRU_dense_reshape = BatchNormalization(axis=2, input_shape=(1, 1*time_num))(output_GRU_dense_reshape)
        output2 = concatenate([std_output_GRU_dense_reshape, input2_], axis=1)
        input1 = tf.concat([_input1_, output2], 2)
        _input1_ = input1[:, 0:2, time_num:]


    output_ = _input1_[:, 0:1, -n_steps_out*time_num:]
    model = Model(inputs=[__input1_, _input2_], outputs=[output_])
    model.compile(optimizer='Adam', loss='mse')
    # model.summary()

    # 定义batch_size大小,应在训练速度足够快的条件下,选择尽量大的batch_size
    # batch_size = int(data_x.shape[0]/5)
    batch_size = 2000
    std_x_data = np.concatenate((std_x_steps_train, std_x_working_train), axis=1)
    if os.path.exists(weight_save_path):
        print(weight_save_path+' has exist! ')
    else:
        # 输入预测函数
        model.fit([std_x_data, std_y_working_train], [std_y_steps_train], batch_size=batch_size, epochs=epochs, verbose=1)
        # 对模型进行存储
        model.save_weights(weight_save_path)
        print("The model has been saved in "+weight_save_path)
    model.load_weights(weight_save_path)
    return model

# 定义函数用于预测参数
# 函数输入：(GRU_model, std_x_steps_test, std_x_working_test, std_y_working_test, std_y_steps_test, std_y_working)
# 函数输出: smape_val, rmse_val
def predictData(GRU_model, std_x_steps_test, std_x_working_test, std_y_working_test, std_y_steps_test, std):
    std_x_data_test = np.concatenate((std_x_steps_test, std_x_working_test), axis=1)
    # 预测数据
    trainPredict_std = GRU_model.predict([std_x_data_test, std_y_working_test])

    trainPredict_std = trainPredict_std.reshape(-1, n_steps_out*time_num)
    std_y_steps_test = std_y_steps_test.reshape(-1, n_steps_out*time_num)

    trainPredict = std.inverse_transform(trainPredict_std).reshape(-1, 1, n_steps_out*time_num)
    data_y_test = std.inverse_transform(std_y_steps_test).reshape(-1)

    # 对有问题的数据（后一时刻步数小于前一时刻的进行处理）
    trainPredict = trainPredict.reshape(-1, time_num)
    for i in range(trainPredict.shape[0]):
        for j in range(1, trainPredict.shape[1]):
            trainPredict[i][j] = trainPredict[i][j] if trainPredict[i][j] >= trainPredict[i][j-1] else trainPredict[i][j-1]
            if j < 5:
                trainPredict[i][j] = 0
    trainPredict = trainPredict.reshape(-1)
    
    trainPredict = trainPredict.reshape(-1)
    data_results = pd.DataFrame(columns=['true', 'predict', 'SMAPE'])
    data_results['true'] = data_y_test
    data_results['predict'] = trainPredict
    data_results['predict'] = data_results['predict'].apply(lambda x: x if x > 0 else 0)
    for i in range(int(data_results.shape[0]/24)):
        data_results['predict'].iloc[i*24] = 0
    data_results['SMAPE'] = 2*abs(data_results['true']-data_results['predict'])/(data_results['predict']+1+data_results['true'])
    data_results.to_csv("./results.csv")
    print(data_results)

    # plt.plot(data_results['true'], label='True steps')
    # plt.plot(data_results['predict'], label = 'Predict steps')
    # plt.legend()
    # plt.show()
    data_y_test = data_results['true'].values
    trainPredict = data_results['predict'].values

    smape_val = smape(trainPredict, data_y_test)
    rmse_val = rmse(trainPredict, data_y_test)
    r2 = r2_score(data_y_test, trainPredict)
    return smape_val, rmse_val, r2

# 定义函数用于预测参数
# 函数输入：(GRU_model, data_x_test, data_y_test ,std)
# 函数输出: smape_val, rmse_val
def predictDataEvery(GRU_model, x_steps, x_week, data_y_week, data_y_steps):
    std_x_steps = StandardScaler()
    std_y_steps = StandardScaler()
    std_x_week = StandardScaler()
    std_y_week = StandardScaler()
    x_steps_std = std_x_steps.fit_transform(x_steps)
    y_steps_std = std_y_steps.fit_transform(data_y_steps)
    x_week_std = std_x_week.fit_transform(x_week)
    y_week_std = std_y_week.fit_transform(data_y_week)
    x_steps_std = x_steps_std.reshape(-1, 1, n_steps_in*time_num)
    x_week_std = x_week_std.reshape(-1, 1, n_steps_in*time_num)
    y_steps_std = y_steps_std.reshape(-1, n_steps_out*time_num)
    y_week_std = y_week_std.reshape(-1, 1, time_num*n_steps_out)
    std_x_data = np.concatenate((x_steps_std, x_week_std), axis=1)
    # 预测数据
    trainPredict_std = GRU_model.predict([std_x_data, y_week_std])
    trainPredict_std = trainPredict_std.reshape(-1, n_steps_out*time_num)

    trainPredict = std_y_steps.inverse_transform(trainPredict_std).reshape(-1, 1, n_steps_out*time_num)
    data_y_steps = std_y_steps.inverse_transform(y_steps_std).reshape(-1)

    # 对有问题的数据（后一时刻步数小于前一时刻的进行处理）
    trainPredict = trainPredict.reshape(-1, time_num)
    for i in range(trainPredict.shape[0]):
        for j in range(1, trainPredict.shape[1]):
            trainPredict[i][j] = trainPredict[i][j] if trainPredict[i][j] >= trainPredict[i][j-1] else trainPredict[i][j-1]
            if j < 5:
                trainPredict[i][j] = 0
    trainPredict = trainPredict.reshape(-1)
    
    trainPredict = trainPredict.reshape(-1)
    data_results = pd.DataFrame(columns=['true', 'predict'])
    data_results['true'] = data_y_steps
    data_results['predict'] = trainPredict
    data_results['predict'] = data_results['predict'].apply(lambda x: x if x > 0 else 0)
    for i in range(int(data_results.shape[0]/24)):
        data_results['predict'].iloc[i*24] = 0

    data_y_test = data_results['true'].values
    trainPredict = data_results['predict'].values

    smape_val = smape(trainPredict, data_y_test)
    rmse_val = rmse(trainPredict, data_y_test)
    r2 = r2_score(data_y_test, trainPredict)
    r = calc_corr(data_y_test, trainPredict)
    return smape_val, rmse_val, r2, r

if __name__ == "__main__":
    # 导入数据
    filePath_save_x_data = "./XY/mergeXY/merge_x_data.npy"
    filePath_save_y_working = "./XY/mergeXY/merge_y_working.npy"
    filePath_save_y_steps = "./XY/mergeXY/merge_y_steps.npy"

    x_data = np.load(filePath_save_x_data)
    y_working = np.load(filePath_save_y_working).reshape(-1, time_num*n_steps_out)
    y_steps = np.load(filePath_save_y_steps).reshape(-1, time_num*n_steps_out)

    x_steps = []
    x_working = []

    # 要分解出x_steps和x_working
    for i in range(x_data.shape[0]):
        x_steps.append(x_data[i][0])
        x_working.append(x_data[i][1])
    x_steps = np.array(x_steps)
    x_working = np.array(x_working)
    
    # 要对步数和周数数据进行标准化
    std_x_steps = StandardScaler()
    std_y_steps = StandardScaler()
    std_x_working = StandardScaler()
    std_y_working = StandardScaler()
    x_steps_std = std_x_steps.fit_transform(x_steps)
    y_steps_std = std_y_steps.fit_transform(y_steps)
    x_working_std = std_x_working.fit_transform(x_working)
    y_working_std = std_y_working.fit_transform(y_working)
    x_steps_std = x_steps_std.reshape(-1, 1, n_steps_in*time_num)
    x_working_std = x_working_std.reshape(-1, 1, n_steps_in*time_num)
    y_steps_std = y_steps_std.reshape(-1, 1, n_steps_out*time_num)
    y_working_std = y_working_std.reshape(-1, 1, time_num*n_steps_out)
    

    # 分割数据集为测试集和训练集
    std_x_steps_train, std_x_steps_test, std_x_working_train, std_x_working_test, std_y_working_train, std_y_working_test, std_y_steps_train, std_y_steps_test = train_test_split(x_steps_std, x_working_std, y_working_std, y_steps_std, test_size=0.2, random_state=1)

    # # 输入预测函数
    # trainModel(std_x_steps_train, std_x_working_train, std_y_working_train, std_y_steps_train)


    # if os.path.exists(weight_save_path):
    #     print(weight_save_path+' has exist! ')
    # else:
    #     # 输入预测函数
    #     trainModel(std_x_steps_train, std_x_working_train, std_y_working_train, std_y_steps_train)
    GRU_model = trainModel(std_x_steps_train, std_x_working_train, std_y_working_train, std_y_steps_train)
    # 加载预测模型
    
    smape_val, rmse_val, r2 = predictData(GRU_model, std_x_steps_test, std_x_working_test, std_y_working_test, std_y_steps_test, std_y_steps)
    print("The SMAPE is "+str(smape_val*100)+"%. ")
    print("The rmse is "+str(rmse_val))
    print("The r2 score is "+str(r2))

    filesPath_x = "./XY/x_data/"
    filesPath_y_working = "./XY/y_working/"
    filesPath_y_steps = "./XY/y_steps/"
    files_x = os.listdir(filesPath_x)
    files_y_working = os.listdir(filesPath_y_working)
    files_y_steps = os.listdir(filesPath_y_steps)
    result_df = pd.DataFrame(columns=['ID', 'SMAPE', 'RMSE', 'R2', 'R'])
    for i in range(len(files_x)):
        file_x = filesPath_x+files_x[i]
        file_y_working = filesPath_y_working+files_y_working[i]
        file_y_steps = filesPath_y_steps+files_y_steps[i]
        file_name = files_x[i].replace('_x_data.npy', '')
        data_x = np.load(file_x)
        x_steps = []
        x_working = []
        # 要分解出x_steps和x_working
        for i in range(data_x.shape[0]):
            x_steps.append(data_x[i][0])
            x_working.append(data_x[i][1])
        x_steps = np.array(x_steps).reshape(-1, n_steps_in*time_num)
        x_working = np.array(x_working).reshape(-1, n_steps_in*time_num)
        data_y_working = np.load(file_y_working).reshape(-1, n_steps_out*time_num)
        data_y_steps = np.load(file_y_steps).reshape(-1, n_steps_out*time_num)
        smape_val, rmse_val, r2, r = predictDataEvery(GRU_model, x_steps, x_working, data_y_working, data_y_steps)
        result_df = result_df.append({'ID':file_name, 'SMAPE':smape_val, "RMSE":rmse_val, "R2":r2, "R":r}, ignore_index=True)
        # print(result_df)
    result_df.to_csv("results_every_ID_withworkingday.csv", index=False)