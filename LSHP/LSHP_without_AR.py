import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from data_utils import *
scale_val = 1.0
writer=SummaryWriter(log_dir='./log')
class LSHP(nn.Module):
    def __init__(self) -> None:
        super(LSHP, self).__init__()
        self.data = DataUtils()
        self.Conv1 = nn.Conv2d(21, 21, kernel_size=(3, 1), stride=(3, 1), bias=True)
        self.RNN1 = nn.GRU(n_steps_in*time_num, time_num, 2, batch_first=True, dropout=0.2)
        self.RNN2 = nn.GRU(n_steps_in, 1, 2, batch_first=True, dropout=0.2)
        self.Conv2 = nn.Conv2d(1, 1, kernel_size=(4, 1), stride=(4, 1), bias=True)
        self.linear1 = nn.Linear(n_steps_in, 1)
        self.Conv3 = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=(2, 1), bias=True)

        self.norm1 = nn.BatchNorm1d(n_steps_in)
        self.norm2 = nn.BatchNorm1d(1)
        self.norm3 = nn.BatchNorm1d(time_num)
        self.norm4 = nn.BatchNorm2d(1)
        self.norm5 = nn.BatchNorm2d(1)
        self.norm6 = nn.BatchNorm1d(1)

    def forward(self, Conv1_input, dataYweek_std, dataYhour, linear_input):
        results = torch.zeros(0)
        for day in range(n_steps_out):
            dataYweek = dataYweek_std[:, day, :].reshape(-1, 1, 1, time_num)

            conv1_out = torch.tanh(torch.squeeze(self.Conv1(Conv1_input)))
            conv1_out = self.norm1(conv1_out)

            rnn1 = conv1_out.reshape(-1, 1, n_steps_in*time_num)
            rnn2 = conv1_out.permute(0, 2, 1)

            rnn1_out, _ = self.RNN1(rnn1)
            rnn1_out = self.norm2(rnn1_out)
            rnn1_out = torch.tanh(rnn1_out).reshape(-1, 1, 1, time_num)
            
            rnn2_out, __ = self.RNN2(rnn2)
            rnn2_out = self.norm3(rnn2_out)
            rnn2_out = torch.tanh(rnn2_out).reshape(-1, 1, 1, time_num)
            
            conv2_input = torch.cat((rnn1_out, rnn2_out, dataYweek, dataYhour), 2).to(torch.float32)
            conv2_out = torch.tanh(self.Conv2(conv2_input))
            conv2_out = self.norm4(conv2_out)

            # linear1_out = torch.tanh(self.linear1(linear_input)).permute(0, 2, 1).reshape(-1, 1, 1, time_num)
            # linear1_out = self.norm5(linear1_out)

            # # 卷积连接
            # conv3_input = torch.cat((conv2_out, linear1_out), 2)
            # results_day = torch.tanh(self.Conv3(conv3_input)).squeeze(1)

            results_day = conv2_out.reshape(-1, 1, time_num)
            results = torch.cat((results, results_day), 1)
            steps_day = results_day.reshape(-1, 1, 1, time_num)
            week_day = dataYweek
            hour_day = dataYhour

            next_day = torch.cat((steps_day, week_day, hour_day), 2).to(torch.float32)
            Conv1_input = torch.cat((Conv1_input, next_day), 1)[:, 1:, :, :]
        # print(results.shape)
        
        return results

def train(lshp, input_Conv1, dataYweek, dataYhour, linear_input, y_label):
    lshp.train()
    MSEloss = nn.L1Loss()

    # optimizer = torch.optim.Adam(lshp.parameters(), lr=0.01)
    optimizer_1 = torch.optim.Adam(lshp.parameters(), lr = 0.05)
    scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch/50+1))

    # for epoch in range(1, 11):
    #     # train

    #     optimizer_1.zero_grad()
    #     optimizer_1.step()
    #     print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    #     scheduler_1.step()

    epochs = 500
    for epoch in range(epochs):
        y=lshp(input_Conv1, dataYweek, dataYhour, linear_input)
        lshp.zero_grad()
        loss = MSEloss(y, y_label)
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer_1.step()
        scheduler_1.step()
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer_1.param_groups[0]['lr']))
        optimizer_1.zero_grad()
        print(epoch+1, '/', epochs, ':', loss.item())
    save_path = "./model/"
    mkdir(save_path)
    torch.save(lshp.state_dict(), save_path+"lshp-without-tcn-"+str(scale_val)+".pt")
    print("save done")

def eval_RMSE(lshp, input_Conv1, dataYweek, dataYhour, linear_input, y_label):
    lshp.eval()
    MSEloss = nn.MSELoss()
    y=lshp(input_Conv1, dataYweek, dataYhour, linear_input)
    y = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y.reshape(-1, time_num*n_steps_out).detach().numpy()))
    y_label = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y_label.reshape(-1, time_num*n_steps_out).detach().numpy()))
    loss = MSEloss(y, y_label)
    # print(loss.item())
    print("RMSE", math.sqrt(loss.item()))

def eval_L1(lshp, input_Conv1, dataYweek, dataYhour, linear_input, y_label):
    lshp.eval()
    MSEloss = nn.L1Loss()
    y=lshp(input_Conv1, dataYweek, dataYhour, linear_input)
    y = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y.reshape(-1, time_num*n_steps_out).detach().numpy()))
    y_label = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y_label.reshape(-1, time_num*n_steps_out).detach().numpy()))
    loss = MSEloss(y, y_label)
    print("L1 loss", loss.item())
    # print(math.sqrt(loss.item()))

from sklearn.metrics import r2_score
def eval_r2(lshp, input_Conv1, dataYweek, dataYhour, linear_input, y_label):
    lshp.eval()
    y=lshp(input_Conv1, dataYweek, dataYhour, linear_input)
    y = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y.reshape(-1, time_num*n_steps_out).detach().numpy())).reshape(-1)
    y_label = torch.from_numpy(lshp.data.dataYstepsstd.inverse_transform(y_label.reshape(-1, time_num*n_steps_out).detach().numpy())).reshape(-1)
    r2 = r2_score(y_label, y)
    print("R2 score", r2)




from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    save_path = "./model/"
    filepath = save_path+"lshp-without-tcn-"+str(scale_val)+".pt"
    lshp = LSHP()

    input_Conv1 = Variable(torch.from_numpy(lshp.data.dataX).to(torch.float32))
    # print(input_Conv1.shape)
    dataYweek = torch.from_numpy(lshp.data.dataYweek_std)
    dataYhour = torch.from_numpy(np.array(lshp.data.dataYhour_std.tolist()*input_Conv1.shape[0]).reshape(-1, 1, 1, time_num))
    linear_input = Variable(torch.from_numpy(lshp.data.dataXsteps_std.reshape(-1, n_steps_in, time_num)).to(torch.float32)).permute(0, 2, 1)
    y_label = torch.from_numpy(lshp.data.dataYsteps_std).to(torch.float32)

    input_Conv1_train, input_Conv1_test, \
    dataYweek_train, dataYweek_test, \
    dataYhour_train, dataYhour_test, \
    linear_input_train, linear_input_test, \
    y_label_train, y_label_test = \
                                    train_test_split(input_Conv1, \
                                                    dataYweek, \
                                                    dataYhour, \
                                                    linear_input, \
                                                    y_label, \
                                                    test_size=0.2, \
                                                    random_state=1)

    if os.path.exists(filepath):
        lshp1 = LSHP()
        lshp1.load_state_dict(torch.load(filepath))
        print(lshp1)
        eval_RMSE(lshp1, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)
        eval_L1(lshp1, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)
        eval_r2(lshp1, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)
        # import torch
        # from torchviz import make_dot
        # y=lshp1(input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test)
        # # , params=dict(list(lshp1.named_parameters()) + [('input_Conv1_test', input_Conv1_test)] + [('dataYweek_test', dataYweek_test)] + [('dataYhour_test', dataYhour_test)] + [('linear_input_test', linear_input_test)])
        # g = make_dot(y, params=dict(list(lshp1.named_parameters())))
        # g.format='png'
        # g.directory='./'
        # g.view()
        # print(list(lshp1.named_parameters()))
        # import hiddenlayer as h
        # y=lshp1(input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test)
        # vis_graph = h.build_graph(y, torch.zeros(size=[1,1,28,28]))   # 获取绘制图像的对象
        # vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
        # vis_graph.save("./demo1.png")   # 保存图像的路径

    else:
        train(lshp, input_Conv1_train, dataYweek_train, dataYhour_train, linear_input_train, y_label_train)
        eval_RMSE(lshp, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)
        eval_L1(lshp, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)
        eval_r2(lshp, input_Conv1_test, dataYweek_test, dataYhour_test, linear_input_test, y_label_test)


