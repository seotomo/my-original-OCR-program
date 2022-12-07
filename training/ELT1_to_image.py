#ETL1データを画像化するプログラム
import struct
import numpy as np
import os
from PIL import Image
import cv2

#files: ETL1データのファイル名
files = ["ETL1C_"+str(f'{i:02}') for i in range(1,14)]
RECORD_SIZE = 2052
MARGIN = 10

def normalize(curve):
    start = 0
    end = 0

    for i in range(len(curve)):
        if curve[i]>0:
            start = i
            break

    for i in range(len(curve)-1,-1,-1):
        if curve[i]>0:
            end = i
            break

    return start, end



for file in files:
    print(file)
    i = 0
    print("Reading {}".format(file))
    with open(file, 'rb') as f:
        while True:
            s = f.read(RECORD_SIZE)
            if s is None or len(s) < RECORD_SIZE:
                break
            r = struct.unpack(">H2sHBBBBBBIHHHHBBBBHH2016s4x", s)
            img = Image.frombytes('F', (64, 63), r[20], 'bit', (4, 0))
            img = img.convert('L')
            img = img.point(lambda x: 255 - (x << 4))
            i = i + 1
            dirname = r[1].decode('utf-8')
            dirname = dirname.replace('\0', '')
            dirname = dirname.replace(' ', '')
            dirname = dirname.replace('\\', 'YEN')
            dirname = dirname.replace('+', 'PLUS')
            dirname = dirname.replace('-', 'MINUS')
            dirname = dirname.replace('*', 'ASTERISK')

            #読み込んだ画像の前処理
            img = np.array(img)
            ret, img2 = cv2.threshold(img,100,255,cv2.THRESH_OTSU)
            img2 = cv2.bitwise_not(img2)

            #文字を画像中心に持ってきて画像サイズを64×64にリサイズ
            tmp = np.zeros((img2.shape[0]+2, img2.shape[1]+2),np.uint8)
            tmp[1:tmp.shape[0]-1, 1:tmp.shape[1]-1] = img2[:,:]
            x_curve = np.sum(tmp, axis = 0)
            y_curve = np.sum(tmp, axis = 1)

            x_start, x_end = normalize(x_curve)
            y_start, y_end = normalize(y_curve)

            width1 =x_end - x_start
            width2 = y_end - y_start
            longer_width = width1
            
            if width2>width1:
                longer_width = width2
            
            center_x = (width1)//2 + x_start
            center_y = (width2)//2 + y_start

            img2 = img[y_start-1:y_end-1, x_start-1:x_end-1]
            tmp2 = np.zeros((longer_width+MARGIN, longer_width+MARGIN),np.uint8)
            center = (longer_width+MARGIN)//2
            np.putmask(tmp2,tmp2 == 0, 255)
            tmp2[center-(width2//2):center+(width2-(width2//2)),center-(width1//2):center+(width1-(width1//2))] = img2[:,:]
            img2 = cv2.resize(tmp2,(64,64))
            img2 = cv2.bitwise_not(img2)

            os.makedirs(f"ETL1_ETL6/{dirname}",exist_ok=True)
    
            imagefile = f"ETL1_ETL6/{dirname}/{file}_{i:0>6}.png"
            print(imagefile)
            cv2.imwrite(imagefile, img2)