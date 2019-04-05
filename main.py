# torch
import torch
from PIL import Image
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.nn.functional import normalize

classes = ['human', 'beach', 'architecture', 'car', 'dinosaur', 'elephent', 'flower', 'horse', 'mountain',
           'food']


class DataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.batch_size = batch_size
        self.file_names = os.listdir(path)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        pic_index = self.file_names[index].split('.')[0]
        pic_index = int(pic_index)
        im = Image.open(self.path + '/%d.jpg' % pic_index)
        H, W = im.size
        if H != 256:
            im = im.rotate(-90, expand=True)
        im = torchvision.transforms.functional.resize(im, size=(96, 64))
        # print(self.classes[int(pic_index / 100)])
        # im.show()  # 查看图片
        im = torchvision.transforms.functional.to_tensor(im)  # [3,96,64]
        label = torch.Tensor([int(pic_index / 100)]).long()
        return im, label


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        # [batch_size,16,48,32]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        # [batch_size,32,24,16]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [batch_size,64,12,8]
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(12, 8))
        self.relu4 = nn.ReLU()
        # [batch_size,128,1,1]
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.relu5 = nn.ReLU()
        # [batch_size,64,1,1]
        self.linear1 = nn.Linear(64, 10)
        self.net = nn.Sequential(self.conv1, self.relu1, self.max_pool1, self.conv2, self.relu2, self.max_pool2,
                                 self.conv3, self.relu3, self.max_pool3, self.conv4, self.relu4, self.conv5, self.relu5)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu1(input)
        input = self.max_pool1(input)
        input = self.conv2(input)
        input = self.relu2(input)
        input = self.max_pool2(input)
        input = self.conv3(input)
        input = self.relu3(input)
        input = self.max_pool3(input)
        input = self.conv4(input)
        input = self.relu4(input)
        input = self.conv5(input)
        output = self.relu5(input)
        output = self.linear1(output.view(-1, 64))
        output = torch.sigmoid(output)
        output = torch.softmax(output, dim=1)
        return output


class Trainer(object):
    def __init__(self, train_dataloader, test_dataloader, model, epoch=20000, lr=0.0001):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cross_entropy = nn.CrossEntropyLoss()

    def start(self):
        self.epoch_losses = []
        self.epoch_test_losses = []
        self.epoch_accs = []
        self.epoch_test_accs = []
        for i in range(1, self.epoch + 1):
            epoch_loss = 0
            epoch_test_loss = 0
            epoch_acc = 0
            epoch_test_acc = 0
            # 训练一个epoch
            for batch_data in self.train_dataloader:
                # 训练一个batch
                im, label = batch_data
                im, label = im.to(self.device), label.to(self.device).view(-1)
                result = self.train_batch(im, label)
                epoch_loss += result[0]
                predict_output = result[1]
                epoch_acc += self.get_acc(predict_output=predict_output, label=label)
            epoch_loss /= len(self.train_dataloader)
            epoch_acc /= len(self.train_dataloader)
            self.epoch_losses.append(epoch_loss)
            self.epoch_accs.append(epoch_acc)
            # 每训练一个epoch，测试一次
            epoch_test_loss, epoch_test_acc = self.test(save_result=(i == self.epoch))
            self.epoch_test_losses.append(epoch_test_loss)
            self.epoch_test_accs.append(epoch_test_acc)
            print("epoch %d:train loss %f,test loss %f,train acc %f,test acc %f" % (
                i, epoch_loss, epoch_test_loss, epoch_acc, epoch_test_acc))

    def get_acc(self, predict_output, label):
        with torch.no_grad():
            test_num = label.size()[0]
            max_index = predict_output.max(dim=1)[1]
            count = (max_index == label).sum().item()
            return count / test_num

    def train_batch(self, img, label):
        self.optimizer.zero_grad()
        # 训练一个batch
        predict_output = self.model(img)  # +
        batch_loss = self.cross_entropy(predict_output, label)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item(), predict_output
        # epoch_acc += self.get_acc(predict_output=predict_output, label=label)

    def test(self, save_result=False):
        epoch_test_loss = 0
        epoch_test_acc = 0
        predict_output = None
        with torch.no_grad():
            for img, label in self.test_dataloader:
                img, label = img.to(self.device), label.to(self.device).view(-1)
                predict_output = self.model(img)
                batch_loss = self.cross_entropy(predict_output, label)
                epoch_test_loss += batch_loss.item()
                epoch_test_acc += self.get_acc(predict_output=predict_output, label=label)
            epoch_test_loss /= len(self.test_dataloader)
            epoch_test_acc /= len(self.test_dataloader)
        if save_result:
            self.predict_output = predict_output
        return epoch_test_loss, epoch_test_acc

    def draw(self):
        # 得到预测结果
        max_index = self.predict_output.max(dim=1)[1]
        test_num = max_index.size()[0]
        # [200,1]
        label = [item for item in self.test_dataloader][0][1]
        # [200,1]
        matrix = torch.tensor((), dtype=torch.float64)
        matrix = matrix.new_zeros((10, 10))
        # 统计
        for i in range(test_num):
            matrix[label[i], max_index[i]] += 1
        # 绘制混淆矩阵，纵坐标真实Label，横坐标预测Label
        print(matrix)
        matrix = normalize(matrix,dim=1).view(10,10).numpy()
        print(matrix)
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        # 设置坐标轴刻度数量
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        # 设置坐标轴刻度名字
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        # 横坐标刻度名字倾斜45度
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, str(matrix[i, j])[0:4], ha="center", va="center", color="w")
        ax.set_title("matrix")
        fig.tight_layout()
        fig.colorbar(im, ax=ax)
        plt.savefig('result/matrix.jpg')


if __name__ == "__main__":
    # 准备数据
    batch_size = 800
    train_data = DataSet(path='train')
    test_data = DataSet(path='test')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, pin_memory=True)
    # 准备模型
    model = VGG()
    print("有%d个参数" % sum(p.numel() for p in model.parameters()))
    # 开始训练
    trainer = Trainer(train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, epoch=1000,
                      lr=0.001)
    trainer.start()
    trainer.draw()
