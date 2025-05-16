from audioset_models import *
import time
import torch.nn as nn
from torch import optim
import numpy as np
import os
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score
from openpyxl import Workbook

# 换成audioset_models里的模型
model = Res1dNet31(sample_rate=44100, window_size=1024, mel_bins=64, hop_size=320, fmin=50, classes_num=527,fmax=14000)
# model.load_state_dict(torch.load(r'E:\python\pycharm\pytorch\model_fuwuqi\audioset_tagging_cnn-master\pytorch\Res1dNet51_mAP=0.355.pth')['model'])
classes_num = 2
model.fc_audioset = nn.Linear(2048,2)


model = torch.load(r'E:\python\pycharm\pytorch\model\audioset_tagging_cnn-master\pytorch\Res1dNet31_1.5s_aug1')

class AudioDataset(Dataset):

    def __init__(self, root_dir, target_length):
        self.root_dir = root_dir
        self.target_length = target_length
        self.classes = os.listdir(root_dir)
        self.data = []

        for i, label in enumerate(self.classes):
            class_dir = os.path.join(root_dir, label)
            files = os.listdir(class_dir)
            for file in files:
                file_path = os.path.join(class_dir, file)
                self.data.append((file_path, i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        desired_duration = 10.0
        orig_audio, sr = librosa.load(file_path, sr=None)
        orig_audio_len = librosa.get_duration(y=orig_audio, sr=sr)
        # 计算需要复制的次数
        num_repeats = int(np.ceil(10 / orig_audio_len))

        # 将音频数据重复复制并截取，使其长度精确为 10 秒
        extended_audio = np.tile(orig_audio, num_repeats)[:int(10 * sr)]

        return extended_audio, label, file_path


train_dir = './datasets/1/train'
test_dir = './test_datasets'
train_dataset = AudioDataset(train_dir, 441000)
test_dataset = AudioDataset(test_dir, 441000)

train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)



#
def train(model, optimizer, criterion, train_loader, device):
    total_loss = 0
    total_correct = 0

    # 将模型置为训练模式
    model.train()

    # 遍历训练集
    for data, target, file_path in train_loader:

        data, target = data.to(device), target.to(device)
        data = torch.squeeze(data)

        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        output = [output['clipwise_output']]
        output = torch.stack(output, dim=0)
        output = torch.squeeze(output, dim=0)

        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 统计损失
        total_loss += loss.item()
        # 计算准确率
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    # 计算准确率
    accuracy = total_correct / len(train_loader.dataset)

    return avg_loss, accuracy


def ceshi(model, criterion, test_loader, device):
    total_loss = 0
    total_correct = 0
    y_true = []
    y_pred = []
    noncough_correct = 0
    noncough_total = 0
    model.eval()
    score_address = []
    score = []
    start_time = time.time()
    with torch.no_grad():
        # 遍历测试集

        for data, target, file_path in test_loader:
            # 将数据移到设备上
            data, target = data.to(device), target.to(device)
            # 前向传播
            output = model(data)

            output = [output['clipwise_output']]
            output = torch.stack(output, dim=0)
            output = torch.squeeze(output, dim=0)
            output_list = output.tolist()

            score.append(output_list)
            score_address.append(file_path)
            # 计算损失
            loss = criterion(output, target)
            # 统计损失
            total_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()

            y_true.extend(target.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

            # 统计noncough类的准确率
            noncough_predicted = predicted[target == 1]
            noncough_total += noncough_predicted.size(0)
            noncough_correct += (noncough_predicted == 1).sum().item()
            result = [item for sublist in score_address for item in sublist]

            score_result = [item for sublist in score for item in sublist]

            result_num = []

            for path in result:
                # 利用字符串方法提取数字部分
                number = path.split('.')[-2].split('\\')[-1]
                result_num.append(number)

            print(result_num)
            filepath1 = r"D:\数据增强实验\score2.xlsx"
            filepath2 = r"D:\数据增强实验\address2.xlsx"

            # 创建一个新的 Excel 工作簿
            workbook = Workbook()
            # 选择默认的活动工作表
            sheet = workbook.active

            # 将数据写入表格
            for row in score_result:
                sheet.append(row)

            # 保存工作簿到指定路径的文件
            workbook.save(filepath1)
            # 创建一个新的 Excel 工作簿
            workbook = Workbook()
            # 选择默认的活动工作表
            sheet = workbook.active
            # 将数据写入表格
            for row in result_num:
                sheet.append([row])

            # 保存工作簿到指定路径的文件
            workbook.save(filepath2)
    end_time = time.time()  # 结束计时
    execution_time = (end_time - start_time) * 1000  # 计算执行时间（毫秒）

    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    # 计算准确率
    accuracy = total_correct / len(test_loader.dataset)
    # 计算F1分数
    f1 = f1_score(y_true, y_pred,pos_label=0)
    accuracy_true = accuracy_score(y_true, y_pred)
    # 计算召回率
    recall = recall_score(y_true, y_pred, pos_label=0)
    # 计算精确度
    precision = precision_score(y_true, y_pred, pos_label=0)
    # 计算noncough类准确率
    noncough_accuracy = noncough_correct / noncough_total

    return avg_loss, accuracy,f1, recall, precision,noncough_accuracy,accuracy_true,execution_time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 70
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, optimizer, criterion, train_dataloader, device)
    # val_loss, val_accuracy = validate(model, criterion, val_loader, device)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train accuracy:{train_accuracy:.4f}")
# 测试
test_loss, test_accuracy, f1, recall, precision, noncough_acc,acc_true ,execution_time= ceshi(model, criterion, test_dataloader, device)
print(f"loss: {test_loss:.4f},accuracy: {test_accuracy:.4f},f1:{f1:.4f}, recall:{recall:.4f}, precision:{precision:.4f},noncough_acc:{noncough_acc:.4f},acc_true:{acc_true:.4f},time:{execution_time:.4f}")
torch.save(model,'./Res1dNet31_aug1')