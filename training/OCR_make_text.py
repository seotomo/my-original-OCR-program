#文字画像データのパスとラベルをテキストデータとして保存するプログラム
import numpy as np
import os
import glob
from tqdm import tqdm
import random

#パス・ラベルの読み込み
train_path = []
validate_path = []
train_label = []
validate_label = []
labels = []


path1 = "emnist"
files1 = os.listdir(path1)
files_dir1 = [f for f in files1 if os.path.isdir(os.path.join(path1, f))]

for i, dir in tqdm(enumerate(files_dir1)):
    filenames = glob.glob(path1 +"\\"+ dir +"\\*.png")

    l1 = [i]*160
    l2 = [i]*40

    labels.append(dir)
    train_path.extend(filenames[:160])
    validate_path.extend(filenames[160:])
    train_label.extend(l1)
    validate_label.extend(l2)


path2 = "ETL1_ETL6"
files2 = os.listdir(path2)
files_dir2 = [f for f in files2 if os.path.isdir(os.path.join(path2, f))]

START1 = len(files_dir1)
print(START1)
for i, dir in tqdm(enumerate(files_dir2, START1)):
    filenames = glob.glob(path2 +"\\"+ dir +"\\*.png")

    l1 = [i]*160
    l2 = [i]*40

    labels.append(dir)
    train_path.extend(filenames[:160])
    validate_path.extend(filenames[160:])
    train_label.extend(l1)
    validate_label.extend(l2)


path3 = "ETL9B"
files3 = os.listdir(path3)
files_dir3 = [f for f in files3 if os.path.isdir(os.path.join(path3, f))]

START2 = len(files_dir1) + len(files_dir2)
for i, dir in tqdm(enumerate(files_dir3, START2)):
    filenames = glob.glob(path3 +"\\"+ dir +"\\*.png")
    
    
    l1 = [i]*160
    l2 = [i]*40

    labels.append(dir)
    train_path.extend(filenames[:160])
    validate_path.extend(filenames[160:])
    train_label.extend(l1)
    validate_label.extend(l2)


#パスとラベルをシャッフル
training = list(zip(train_path, train_label))
validation = list(zip(validate_path, validate_label))
random.shuffle(training)
random.shuffle(validation)
x_train,y_train = zip(*training)
x_validate, y_validate = zip(*validation)


#テキストデータに書き込み
with open("OCR_train_data.txt","w") as f:
    for i,data in enumerate(x_train):
        f.write(data + " " + str(y_train[i]) + "\n")

with open("OCR_validate_data.txt","w") as f:
    for i,data in enumerate(x_validate):
        f.write(data + " " + str(y_validate[i]) + "\n")


#ラベルと文字の対応関係を保存
np.save("labels",np.array(labels))


