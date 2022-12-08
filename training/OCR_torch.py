#文字分類の深層学習プログラム(Efficientnet-b0使用)
import torch 
import torch.nn as nn
from torchinfo import summary
import torch.optim as Optimizer
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#学習条件
BATCH_SIZE = 512
EPOCHS = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASS = 62+59+3036


#カスタムデータセットの作成
class my_dataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
       
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx]
        batch_y = self.labels[idx]
        #ファイルを開く -> (高さ, 幅, RGB)
        batch_x = cv2.imread(batch_x, 0)
        _, batch_x = cv2.threshold(batch_x,100,255,cv2.THRESH_OTSU)
        batch_x = Image.fromarray(batch_x)
        batch_x = batch_x.convert("L").convert("RGB")
        img_transformed = self.transform(batch_x)

        return img_transformed, batch_y

    
def fit(net, optimizer, criterion, num_epochs, train_loader, val_loader,device):

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

    
        #訓練フェーズ
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        #予測フェーズ
        net.eval()
        count = 0

        for inputs, labels in tqdm(val_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            val_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list


#学習グラフの描画関数
def plot_loss_accuracy_graph(train_acc_list,val_acc_list,train_loss_list,val_loss_list):
    plt.plot(train_acc_list, "-D", color="blue", label="train_accuracy", linewidth=2)
    plt.plot(val_acc_list, "-D", color="black", label="val_accuracy", linewidth=2)
    plt.plot(train_loss_list, "-D", color="green", label="train_loss", linewidth=2)
    plt.plot(val_loss_list, "-D", color="red", label="val_loss", linewidth=2)
    plt.title("Learning result")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy and Loss")
    plt.legend(loc="lower left")
    plt.show() 


def main(): 
    #画像の読み込み
    train_file = open("OCR_train_data.txt", "r")
    datalist1 = train_file.readlines()
    train_path = [i.split()[0] for i in datalist1] 
    train_label = [int(i.split()[1]) for i in datalist1] 

    validate_file = open("OCR_validate_data.txt", "r")
    datalist2 = validate_file.readlines()
    validate_path = [i.split()[0] for i in datalist2] 
    validate_label = [int(i.split()[1]) for i in datalist2]


    transform1 = transforms.Compose([
                    #回転・移動・縮小
                    transforms.RandomRotation(degrees=5, fill=0),
                    transforms.RandomAffine(degrees=[0, 0], translate=(0.1, 0.1), scale=(0.8, 1.2), fill=0),
                    #Tensor型に変換
                    transforms.ToTensor(),
                    #色情報の標準化
                    transforms.Normalize(0.5, 0.5)
                ])

    transform2 = transforms.Compose([
                    #Tensor型に変換
                    transforms.ToTensor(),
                    #色情報の標準化
                    transforms.Normalize(0.5, 0.5)
                ])
    

    # Dataset を作成する。
    train_dataset = my_dataset(train_path, train_label,  transform=transform1)
    validate_dataset = my_dataset(validate_path, validate_label,  transform=transform2)
    print(f'学習データ: {len(train_dataset)}件')
    print(f'検証データ: {len(validate_dataset)}件')


    # DataLoader を作成する。
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, num_workers=2)
    #train_dataloader = [*train_dataloader]


    #モデルの作成
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, CLASS)
    summary(model,(BATCH_SIZE,3,64,64))




    #学習処理の方法を設定
    criterion = nn.CrossEntropyLoss()
    optimizer = Optimizer.Adam(model.parameters(),lr=0.0001)

    #学習(verbose:ログ出力の設定)
    loss, acc,val_loss, val_acc = fit(model, optimizer, criterion, EPOCHS, train_dataloader, val_dataloader, device)
    acc = [i*100 for i in acc]
    val_acc = [i*100 for i in val_acc]
    plot_loss_accuracy_graph(acc, val_acc, loss, val_loss)

    #モデルの保存
    torch.save(model.state_dict(), "OCR_pytorch.pth")

    f=open("OCR_learning_curve.txt","w")
    for i in range(EPOCHS):
        f.write(f"EPOCHS:{i+1} accuracy:{acc[i]:.4f}"\
                f" loss:{loss[i]:.4f}"\
                f" val_acc:{val_acc[i]:.4f}"\
                f" val_loss:{val_loss[i]:.4f}\n")


if __name__ == '__main__':
    main()

