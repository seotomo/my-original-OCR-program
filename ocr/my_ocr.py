#OCRプログラム
import statistics
import cv2
import numpy as np
from scipy.signal import find_peaks
import torch 
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


#Sobelフィルタとラベリング処理を組み合わせた直線除去
def RemoveLinesFromImage(gray_image, direction):
    kernel = np.ones((3,3),dtype=np.uint8)
    thresh, _ = cv2.threshold(gray_img, 100, 255, cv2.THRESH_OTSU)
    
    #5回繰り返し,太い線にも対応
    for _ in range(5): 
        _, binary_image = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)
        mask = np.zeros((binary_image.shape[0], binary_image.shape[1]), dtype = np.uint8) 

        #Sobelフィルタの方向指定
        if direction == "horizontal":
            edge_enhanced_image = cv2.Sobel(binary_image,-1,0,1)
        else:
            edge_enhanced_image = cv2.Sobel(binary_image,-1,1,0)

        _, edge_enhanced_image = cv2.threshold(edge_enhanced_image,1,255,cv2.THRESH_BINARY)
        
        #ラベリング
        label_numbers, labelimage, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(edge_enhanced_image, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

        if direction == "horizontal":
            for label in range(label_numbers):
                label_width = data[label][2]
                label_height = data[label][3]

                #ラベルの縦横比が20以上で線と判定
                if label_width//label_height>20:
                     mask[labelimage == label] = 255
        else:
            for label in range(label_numbers):
                label_width = data[label][2]
                label_height = data[label][3]

                #ラベルの縦横比が20以上で線と判定
                if label_height//label_width>20:
                     mask[labelimage == label] = 255
        

        mask = cv2.dilate(mask,kernel, iterations =2)
        gray_image[mask == 255] = 255

    return gray_image

#ノイズ除去
def RemoveNoiseFromImage(gray_image):
    ret, _ = cv2.threshold(gray_image, 100, 255, cv2.THRESH_OTSU)
    ret, binary_image = cv2.threshold(gray_image, ret, 255, cv2.THRESH_TOZERO)
    binary_image = cv2.medianBlur(binary_image, ksize=3)
    ret, binary_image = cv2.threshold(binary_image, 100, 255, cv2.THRESH_OTSU)
    return binary_image


#文字の大きさ情報取得
def GetCharSize(label_numbers, data):

    rect_list = []
    height_list = []
    width_list = []
    for i in range(1,label_numbers):
        rect_list.append(data[i][2]*data[i][3])
        width_list.append(data[i][2])
        height_list.append(data[i][3])
    
    #文字サイズの面積・高さ・幅の中央値を取得
    rect_list = [ rectangle for rectangle in rect_list if rectangle*2 >= np.median(rect_list)]
    width_list = [ width for width in width_list if width*2 >= np.median(width_list)]
    height_list = [ height for height in height_list if height*2 >= np.median(height_list)]

    char_area = int(np.median(rect_list))
    char_height = int(np.median(height_list))
    char_width = int(np.median(width_list))

    return char_area, char_height, char_width


#図形の除去
def RemoveFigure(binary_img, data, label_numbers, char_area):

    FIGURE_RATE = 6
    for i in range(1,label_numbers):
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]

        #文字を囲む四角形の面積が平均の6倍以上で図形と判定
        rect_area = data[i][2]*data[i][3]
        if rect_area < char_area*FIGURE_RATE:
            cv2.rectangle(binary_img, (x0, y0), (x1, y1), 255,-1)
        else:
            cv2.rectangle(binary_img, (x0, y0), (x1, y1), 0,-1)
    
    return binary_img


def FindLinePeaks(binary_image, char_width, char_height, smooth_rate=4):

    kernel =np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0] ], dtype=np.uint8)

    #行認識の前処理
    binary_image = cv2.dilate(binary_image, kernel, iterations = char_width)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        binary_image = cv2.drawContours(binary_image, contours, i, 255,-1)
    binary_image = cv2.erode(binary_image, kernel, iterations = char_width)
 
    line_curve = np.sum(binary_image/255, axis=1)
    for i in range(char_height//smooth_rate+1,img.shape[0]-char_height//smooth_rate-1):
        line_curve[i] = np.mean(line_curve[i-char_height//smooth_rate:i+char_height//smooth_rate])

    line_peaks, _ = find_peaks(line_curve, height=char_height*2)

    return line_peaks

#同一ライン上の文字同士の結合
def ConnectCharsOnLine(binary_image, line_peaks, labelimage, char_area):

    CHAR_AREA_JUDGE_RATE = 3
    for peak in line_peaks:
        line_start = 0
        line_end = 0

        for i in range(labelimage.shape[1]):
            if labelimage[peak,i] > 0:
                forward_value = labelimage[peak,i]
                forward_area = data[forward_value][2] * data[forward_value][3]
                if forward_area * CHAR_AREA_JUDGE_RATE > char_area:
                    line_start = i
                    break

        for i in range(labelimage.shape[1]-1,-1,-1):
            if labelimage[peak,i] > 0:
                back_value = labelimage[peak,i]
                back_area = data[back_value][2] * data[back_value][3]
                if back_area * CHAR_AREA_JUDGE_RATE > char_area:
                    line_end = i
                    break

        cv2.line(binary_image,(line_start,peak),(line_end,peak),255,3)

    return binary_image


def DetectLines(binary_image):
    chararea_dict = {}

    #外接矩形で行の切り出し
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 1
    for contour in contours:
        #外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #文字面積の2倍以上で矩形内の塗りつぶし
        if cv2.contourArea(box)>char_area*2:
            binary_image = cv2.drawContours(binary_image,[box],0,count,-1)    
            chararea_dict[count] = box  
            count+=1


    #同一矩形の認識
    all_linesbox_image = np.zeros((binary_image.shape[0],binary_image.shape[1]),np.uint8)
    all_linesbox_image [(binary_image>0) & (binary_image < 255)] = 255

    contours, _ = cv2.findContours(all_linesbox_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    AREA_MATCH_RATE = 0.3
    for i in range(len(contours)):
        one_linebox_image = np.zeros((binary_image.shape[0],binary_image.shape[1]), np.uint8)
        one_linebox_image = cv2.drawContours(one_linebox_image, contours, i, 255, -1)

        #隣接矩形内の重複しないピクセル値と最頻値を取得
        mode = int(statistics.mode(binary_image[one_linebox_image == 255]))
        values= list(set(binary_image[one_linebox_image == 255]))

        mode_line_image = np.zeros((binary_image.shape[0],binary_image.shape[1]), np.uint8)
        mode_line_image = cv2.drawContours(mode_line_image, [chararea_dict[mode]], 0, 1, -1)
        if len(values)>= 2:
            for value in values:
                if (value != 0) and (value != mode) and (value != 255):
                    value_inline_image = np.zeros((binary_image.shape[0],binary_image.shape[1]), np.uint8)
                    value_inline_image = cv2.drawContours(value_inline_image, [chararea_dict[value]], 0, 1, -1)

                    add_lines_image = cv2.add(mode_line_image, value_inline_image)
                    union = np.count_nonzero(add_lines_image==2)
                    area = np.count_nonzero(value_inline_image==1)
                    #面積の一致度が3割以上で同一と判定
                    if float(union) / area > AREA_MATCH_RATE: 
                        binary_image[value_inline_image == 1] = mode


    kernel =np.array([  [0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0] ], dtype=np.uint8)

    lines_position = {}
    for i in range(1,255):

        line_exist = np.count_nonzero(binary_image == i)
        if line_exist > 0:

            #細かなラベル抜きの行画像
            out_noise_line_image = np.zeros((binary_image.shape[0], binary_image.shape[1]), np.uint8)
            np.putmask(out_noise_line_image, binary_image==i, 255)
            contours1, _ = cv2.findContours(out_noise_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            #細かなラベルを含む行画像
            in_noise_line_image = np.zeros((binary_image.shape[0], binary_image.shape[1]), np.uint8)
            np.putmask(in_noise_line_image,binary_image==i,255)
            np.putmask(in_noise_line_image,binary_image==255,255)
            #近距離で離れた文字の接続
            in_noise_line_image =cv2.dilate(in_noise_line_image,kernel,iterations = round(char_width*2.5))
            in_noise_line_image = cv2.erode(in_noise_line_image, kernel, iterations = round(char_width*2))
            contours2, _ = cv2.findContours(in_noise_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            #contour1がcontour2の内部にあれば
            for contour1 in contours1:
                rect1 = cv2.minAreaRect(contour1)
                center, _, _ = rect1
                
                for contour2 in contours2:

                    inside_judge = cv2.pointPolygonTest(contour2, center, True)
                    if inside_judge > 0:
                        rect2 = cv2.minAreaRect(contour2)
                        line_y_axis = int(rect2[0][1])
                        four_corners = cv2.boxPoints(rect2)
                        four_corners = np.int0(four_corners)
                        binary_image = cv2.drawContours(binary_image, [four_corners], 0, i, -1)  
                        lines_position[line_y_axis] = rect2

    return binary_image, lines_position


def ProcessingLines(lines_position, gray_image):

    height_list = [ int(rect[1][1]) if int(rect[1][1])<int(rect[1][0]) else int(rect[1][0]) for _, rect in lines_position]
    height_list = sorted(height_list)
    height_medium = height_list[len(height_list)//2]   

    lines_list = []
    line_image_list = []
    LINEAREA = (height_medium//2)**2

    #射影変換＋行の位置
    for start_y, rect in lines_position:
        start_x = int(rect[0][0])
        height = int(rect[1][1])
        width = int(rect[1][0])

        if height*width < LINEAREA:
            continue

        if height > width:
            tmp = height
            height = width
            width = tmp
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        tmp_noline_image = np.zeros((binary_img.shape[0], binary_img.shape[1]), np.uint8)
        tmp_noline_image = cv2.drawContours(tmp_noline_image, [box], 0, 255, -1)  

        one_line_image = np.copy(gray_image)
        one_line_image[tmp_noline_image == 0] = 255 
        
        #射影変換
        left = sorted(box,key=lambda x:x[0]) [:2]
        right = sorted(box,key=lambda x:x[0]) [2:]
        left = [ list(i) for i in left]
        right = [ list(i) for i in right]
        left_down= sorted(left,key=lambda x:x[1]) [0]
        left_up= sorted(left,key=lambda x:x[1]) [1]
        right_down= sorted(right,key=lambda x:x[1]) [0]
        right_up= sorted(right,key=lambda x:x[1]) [1]

        input_coordinate = np.float32([[left_down],[left_up],[right_up],[right_down]])
        output_coordinate = np.float32([[0,0],[0,height],[width,height],[width,0]])
        matrix = cv2.getPerspectiveTransform(input_coordinate, output_coordinate)
        one_line_image = cv2.warpPerspective(one_line_image, matrix, (width,height))

        line_image_list.append(one_line_image)


        #同じ行に映りこむ違う行の除去
        kernel =np.array([  [0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0] ], dtype=np.uint8)

        _, one_line_image = cv2.threshold(one_line_image, 1, 255, cv2.THRESH_OTSU)
        multi_linecheck_image = cv2.erode(one_line_image, kernel, iterations = height)
        multi_linecheck_image = cv2.bitwise_not(multi_linecheck_image)

        label_numbers, labelimage, data, _ = cv2.connectedComponentsWithStatsWithAlgorithm(multi_linecheck_image,8,cv2.CV_16U, cv2.CCL_DEFAULT)
        maxim = np.argmax(data[:,4])

        for label in range(label_numbers):
            if (label != maxim):

                label_ycenter = data[label][1]+data[label][3]//2
                if (( label_ycenter < one_line_image.shape[0]//5)) or ((label_ycenter > one_line_image.shape[0]*4//5)):
                    one_line_image[labelimage == label] = 255

        lines_list.append((start_y, start_x, one_line_image))
    return lines_list, line_image_list
    

def TransformLinesList(lines_list1):

    y_axis = 1
    lines_list2 = []
    pre_axis = 0
    for i, line in enumerate(lines_list1):
        start_y, start_x, image = line[0], line[1], line[2]
        line_height = image.shape[0] 

        if i == 0:
            pass
        else:
            if abs(start_y - pre_axis) < line_height//3:
                pass
            else:
                y_axis+=1
        pre_axis = start_y 
        lines_list2.append((y_axis, start_x, image))

    lines_list2.sort(key=lambda x: (x[0], x[1]))

    return lines_list2


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


#深層学習前の画像処理
def PreprocessingForLearning(image):
    MARGIN = 10

    #gray_imgの場合
    #ret, image = cv2.threshold(image, 1,255,cv2.THRESH_OTSU)

    image = cv2.bitwise_not(image)

    tmp = np.zeros((image.shape[0]+2, image.shape[1]+2),np.uint8)
    tmp[1:tmp.shape[0]-1, 1:tmp.shape[1]-1] = image[:,:]
    x_curve = np.sum(tmp, axis = 0)
    y_curve = np.sum(tmp, axis = 1)

    xaxis_start, xaxis_end = normalize(x_curve)
    yaxis_start, yaxis_end = normalize(y_curve)

    width1 =xaxis_end - xaxis_start
    width2 = yaxis_end - yaxis_start
    longer_width = width1
    
    if width2>width1:
        longer_width = width2

    image2 = image[yaxis_start-1:yaxis_end-1, xaxis_start-1:xaxis_end-1]
    tmp3 = np.zeros((longer_width+MARGIN, longer_width+MARGIN),np.uint8)
    center = (longer_width+MARGIN)//2
    tmp3[center-(width2//2):center+(width2-(width2//2)),center-(width1//2):center+(width1-(width1//2))] = image2[:,:]
    #ret, tmp3 = cv2.threshold(tmp3, 1,255,cv2.THRESH_OTSU)
    image2 = cv2.resize(tmp3,(64,64))
    _, image2 = cv2.threshold(image2,1,255,cv2.THRESH_OTSU)

    return image2


#transformedlineslist:2値化イメージ　line_image_list:グレースケール
def DetectChars(transformed_lines_list, line_image_list, data_form):

    #手書きと印刷の係数[手書き, 印刷]
    MARGIN_RATE = [7, 3]
    CONNECT_WIDTH_RATE = [1.0, 1.1]
    SMALL_RECT_SIZE = [0.5, 0.6]
    DIVIDE_SIZE = [1.0, 1.3]
    DIVIDE_MARGIN_RATE = [3, 3]
    DIVIDE_RATE = [0.4, 0.6]

    pre_xaxis = 0
    pre_yaxis = 0
    char_list = []
    for i, line in enumerate(transformed_lines_list):
        start_y, start_x, img = line[0], line[1], line[2]
        gray_img = line_image_list[i]

        img2 = np.copy(img)
        img2[img == 0] = 1
        img2[img == 255] = 0
        HEIGHT = img.shape[0]
        column_curve = list(np.sum(img2, axis = 0))
        column_curve.append(0)
        column_curve.insert(0,0)
        start_list = [x for x in range(len(column_curve)-1) if (column_curve[x] == 0) and (column_curve[x+1]>0) ]
        end_list = [x for x in range(len(column_curve)-1) if (column_curve[x]>0) and (column_curve[x+1] == 0) ]

        
        for j in range(3):
            del_start = []
            del_end = []
            add_list = []

            for k in range(len(start_list)):
                #ノイズ線の除去
                row_curve = list(np.sum(img2[:,start_list[k]:end_list[k]], axis = 1))
                if np.count_nonzero(row_curve) < HEIGHT//10:
                    if (np.argmax(row_curve) < HEIGHT//4) or (np.argmax(row_curve) >HEIGHT*3//4):
                        del_start.append(k)
                        del_end.append(k)

            start_list = [start_list[x] for x in range(len(start_list)) if x not in del_start]
            end_list = [end_list[x] for x in range(len(end_list)) if x not in del_end]
            del_start.clear()
            del_end.clear()

            rect_width = []
            for k in range(len(start_list)):
                if end_list[k] - start_list[k] > 1:
                    rect_width.append(end_list[k] - start_list[k])
                else:
                    del_start.append(k)
                    del_end.append(k)

            start_list = [start_list[x] for x in range(len(start_list)) if x not in del_start]
            end_list = [end_list[x] for x in range(len(end_list)) if x not in del_end]

            del_start.clear()
            del_end.clear()


            tmp_width = sorted(rect_width)
            median_rect_width = tmp_width[len(tmp_width)//2]

            #分裂した文字の接合
            JUDGE_VALUE = [HEIGHT, median_rect_width]
            MARGIN_SPACE = JUDGE_VALUE[data_form]//MARGIN_RATE[data_form]
            for k in range(len(start_list)):

                if k >= 1:
                    pre_margin = start_list[k] - end_list[k-1]
                    pre_width = end_list[k-1] - start_list[k-1]
                if k < len(start_list)-1:
                    later_margin = start_list[k+1] - end_list[k]
                    later_width = end_list[k+1] - start_list[k+1]

                if k == 0:
                    if rect_width[k] <= JUDGE_VALUE[data_form]*SMALL_RECT_SIZE[data_form]:
                        x = round(rect_width[k]/JUDGE_VALUE[data_form], 2)
                        if (later_margin <= MARGIN_SPACE) and (later_width <= JUDGE_VALUE[data_form]*(CONNECT_WIDTH_RATE[data_form]-x)):
                            del_end.append(k)
                            del_start.append(k+1)

                elif (k > 0) and (k < len(start_list)-1):
                    if rect_width[k] <= JUDGE_VALUE[data_form]*SMALL_RECT_SIZE[data_form]:
                        x = round(rect_width[k]/JUDGE_VALUE[data_form], 2)
                        if (pre_margin <= MARGIN_SPACE) and (pre_width <= JUDGE_VALUE[data_form]*(CONNECT_WIDTH_RATE[data_form]-x)):
                            del_end.append(k-1)
                            del_start.append(k)
                            
                        if (later_margin <= MARGIN_SPACE) and (later_width <= JUDGE_VALUE[data_form]*(CONNECT_WIDTH_RATE[data_form]-x)):
                            del_end.append(k)
                            del_start.append(k+1)

                else:
                    if rect_width[k] <= JUDGE_VALUE[data_form]*SMALL_RECT_SIZE[data_form]:
                        x = round(rect_width[k]/JUDGE_VALUE[data_form], 2)
                        if (pre_margin <= MARGIN_SPACE) and (pre_width <= JUDGE_VALUE[data_form]*(CONNECT_WIDTH_RATE[data_form]-x)):
                            del_end.append(k-1)
                            del_start.append(k)

            del_start = sorted(set(del_start))
            del_end = sorted(set(del_end))

            start_list = [start_list[x] for x in range(len(start_list)) if x not in del_start]
            end_list = [end_list[x] for x in range(len(end_list)) if x not in del_end] 


            #つながった文字の分割 
            for k in range(len(start_list)):

                end_point = end_list[k]
                distance = [start_list[k], end_list[k]]
                roop = 1
                while roop > 0:
                    roop = 0
                    number = len(distance)-1
                    
                    for l in range(number):
                        width = distance[l+1] - distance[l]
                        end_point = distance[l+1]
                        if width >= round(HEIGHT*DIVIDE_SIZE[data_form]):
                            x = np.argmin(column_curve[distance[l]+HEIGHT//DIVIDE_MARGIN_RATE[data_form]:end_point-HEIGHT//DIVIDE_MARGIN_RATE[data_form]]) + distance[l]+HEIGHT//DIVIDE_MARGIN_RATE[data_form]

                            while (x - distance[l]) < HEIGHT*DIVIDE_RATE[data_form]:
                                x = np.argmin(column_curve[x+HEIGHT//10:end_point-HEIGHT//DIVIDE_MARGIN_RATE[data_form]]) + x + HEIGHT//10
                            end_point = x          
                                
                            add_list.append(end_point)
                            distance.append(end_point)
                            distance = sorted(distance)

                            roop +=1
                        
                
            start_list.extend(add_list)
            start_list = sorted(start_list)

            end_list.extend(add_list)
            end_list = sorted(end_list)

        #gray_img:グレースケール 消したノイズが入ってくるのでダメ　img:２値化
        if start_y == pre_yaxis:
            for j in range(len(start_list)):
                if data_form == 0:
                    char_img = img[:,start_list[j]:end_list[j]]
                else:
                    char_img = gray_img[:,start_list[j]:end_list[j]]
                char_img = PreprocessingForLearning(char_img)
                char_list.append((start_y, pre_xaxis+j+1, char_img))

        else:
            for j in range(len(start_list)):
                pre_xaxis = 0
                if data_form == 0:
                    char_img = img[:,start_list[j]:end_list[j]]
                else:
                    char_img = gray_img[:,start_list[j]:end_list[j]]
                char_img = PreprocessingForLearning(char_img)
                char_list.append((start_y, pre_xaxis+j+1, char_img))

        pre_xaxis += len(start_list)
        pre_yaxis = start_y

    return char_list
    

kernel = np.ones((3,3),dtype=np.uint8)
kernel2 =np.array([ [0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0] ], dtype=np.uint8)
kernel3 =np.array([ [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0] ], dtype=np.uint8)

"""
文字認識
"""
#テキスト化したい文書画像のパス名
filepath = "画像パス名"
img = cv2.imread(filepath)
gray_img = cv2.imread(filepath, 0)
data_form = int(input("文書の形式（スキャンされた手書きデータ:0  スキャンされた印刷データ:1）: "))

"""
前処理
"""
#直線除去
gray_img = RemoveLinesFromImage(gray_img, "horizontal")
gray_img = RemoveLinesFromImage(gray_img, "vertical")


#ノイズ除去
if data_form == 0:
    denoised_img = RemoveNoiseFromImage(gray_img)
elif data_form == 1:
    _, denoised_img = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY)
    

"""
行の抽出
"""
#ラベリングによるレイアウト解析
binary_img = cv2.bitwise_not(denoised_img)
binary_img = cv2.erode(binary_img, kernel2)
binary_img = cv2.dilate(binary_img, kernel2)
label_numbers, labelimage, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(binary_img, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

#文字サイズの取得
char_area, char_height, char_width = GetCharSize(label_numbers, data)

#図形の除去
binary_img = RemoveFigure(binary_img, data, label_numbers, char_area)

#文字列の位置取得
line_peaks = FindLinePeaks(binary_img, char_width, char_height, smooth_rate=4)

#文字同士の連結
binary_img = ConnectCharsOnLine(binary_img, line_peaks, labelimage, char_area)

#文字列の切り出し
binary_img, lines_position = DetectLines(binary_img)    
lines_position = sorted(lines_position.items())

#射影変換による行の成形
lines_list, line_image_list = ProcessingLines(lines_position, gray_img)

#行データを扱いやすい形に変更
transformed_lines_list = TransformLinesList(lines_list)

"""
文字の抽出
"""
#文字の検出
char_list = DetectChars(transformed_lines_list, line_image_list, data_form)


"""
予測によるテキスト化
"""
#文字認識
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHAR_CLASS = 3157
#モデルの作成
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, CHAR_CLASS)

#学習済みモデルのパス名
model_path = 'OCR_pytorch.pth'
#文字とラベル番号の対応関係を記録した.npyファイルのパス名
labels_path = 'labels.npy'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
labels = np.load(labels_path)

transform = transforms.Compose([
                    # Tensor型に変換する
                    transforms.ToTensor(),
                    # 色情報の標準化をする
                    transforms.Normalize(0.5, 0.5, 0.5)
                ])

model = model.eval()
predict = []
gyou = 1
print("テキスト化を実行中・・・")
with open("output_sentense.txt","w") as f:
    for y_row, x_column, char_image in tqdm(char_list):
        char_image = Image.fromarray(char_image).convert("RGB")
        char_image = transform(char_image).unsqueeze(0)
        # 予測を実施
        output = model(char_image.to(device))
        _, prediction = torch.max(output, 1)
        result = labels[prediction[0].item()]

        a = "1B2442" + str(result) + "1B2842"
   
        b = bytes.fromhex(a)
        b = b.decode("iso-2022-jp")
        
        if y_row != gyou:
            f.write("\n")
        f.write(str(b))    

        gyou = y_row








