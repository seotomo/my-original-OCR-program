#ETL9Bデータを1枚ずつ画像化するプログラム
import struct
from PIL import Image
import os
import numpy as np
import cv2

#files: ETL9Bデータのファイル名
files = ["ETL9B_"+str(i) for i in range(1,6)]
nrecords = 121440
record_size = 576
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


for i, file in enumerate(files):
    for rc_idx in range(nrecords):           
        with open(file, 'rb') as f:
            f.seek((rc_idx + 1) * record_size)
            s = f.read(record_size)
            r = struct.unpack('>2H4s504s64x', s)
            img = Image.frombytes('1', (64, 63), r[3], 'raw')
            img = img.convert('L')
            file_name = 'ETL9B_{}_{}_{}_{}.png'.format((r[0]-1)%20+1, hex(r[1])[-4:], i+1, rc_idx+1)
            dir_name = "ETL9B\\{}".format(hex(r[1])[-4:])

            #読み込んだ画像の前処理
            img = np.array(img)
            img2 = np.copy(img)

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

            img2 = img[y_start-1:y_end-1, x_start-1:x_end-1]
            tmp2 = np.zeros((longer_width+MARGIN, longer_width+MARGIN),np.uint8)
            center = (longer_width+MARGIN)//2
            tmp2[center-(width2//2):center+(width2-(width2//2)),center-(width1//2):center+(width1-(width1//2))] = img2[:,:]
            img2 = cv2.resize(tmp2,(64,64))

            os.makedirs(dir_name, exist_ok=True)
            imagefile = f"{dir_name}/{file_name}"
            print(imagefile)
            cv2.imwrite(imagefile, img2)

