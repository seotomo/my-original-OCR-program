#EMNISTデータを各ラベル200枚ずつ画像化するプログラム
import emnist
import numpy as np
import cv2
import os
from tqdm import tqdm

           
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


MARGIN = 10

#emnistの読みこみ（byclass or balanced）
pre_x_train1, pre_y_train1 = emnist.extract_training_samples("byclass")
#x_test, y_test = emnist.extract_test_samples('byclass')

emnist_char_list = ['2330','2331','2332','2333','2334','2335','2336','2337','2338','2339',
                     '2341','2342','2343','2344','2345','2346','2347','2348','2349','234a',
                     '234b','234c','234d','234e','234f','2350','2351','2352','2353','2354',
                     '2355','2356','2357','2358','2359','235a','2361','2362','2363','2364',
                     '2365','2366','2367','2368','2369','236a','236b','236c','236d','236e',
                     '236f','2370','2371','2372','2373','2374','2375','2376','2377','2378',
                     '2379','237a']


pre_x_train1 = np.array(pre_x_train1)
pre_y_train1 = np.array(pre_y_train1)

#ラベル番号順に並べ替え
sorted_y_train = np.argsort(pre_y_train1)
pre_x_train1 = pre_x_train1[sorted_y_train]
pre_y_train1 = pre_y_train1[sorted_y_train]


#200枚ずつ画像化して保存
EMNIST_CLASS = 62
total_number = 0
x_train = []
y_train = []
x_validate = []
y_validate = []
labels = []

for i in tqdm(range(EMNIST_CLASS)):

    number = list(pre_y_train1).count(i)
    total_number += number
    os.makedirs(f"emnist/{emnist_char_list[i]}",exist_ok=True)

    for j in range(total_number-number, total_number):

        img = np.array(pre_x_train1[j])
        img2 = np.copy(img)
        tmp = np.zeros((img2.shape[0]+2, img2.shape[1]+2),np.uint8)
        tmp[1:tmp.shape[0]-1, 1:tmp.shape[1]-1] = img2[:,:]
        x_curve = np.sum(tmp, axis = 0)
        y_curve = np.sum(tmp, axis = 1)

        x_start, x_end = normalize(x_curve)
        y_start, y_end = normalize(y_curve)

        x_width =x_end - x_start
        y_width = y_end - y_start
        longer_width = x_width

        if y_width>x_width:
            longer_width = y_width
        
        img2 = img[y_start-1:y_end-1, x_start-1:x_end-1]
        tmp3 = np.zeros((longer_width+MARGIN, longer_width+MARGIN),np.uint8)
        center = (longer_width+MARGIN)//2
        tmp3[center-(y_width//2):center+(y_width-(y_width//2)),center-(x_width//2):center+(x_width-(x_width//2))] = img2[:,:]
        img2 = cv2.resize(tmp3,(64,64))
        #ret, img2 = cv2.threshold(img2,1,255,cv2.THRESH_OTSU)
        
        imagefile = f"emnist/{emnist_char_list[i]}/{emnist_char_list[i]}_{j}.png"
        if j < (total_number-number)+200:
            cv2.imwrite(imagefile, img2)
        else:
            break
