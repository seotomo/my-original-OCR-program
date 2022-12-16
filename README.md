# 自作OCRプログラムの作成（詳細版）

## 目的
既存のOCRソフトは数多くあるが、日本語の文字認識は非常に難しいと言われている。理由として①ひらがな・カタカナに似た文字が多い、②「け」や「は」など独立した複数の画から構成される文字が多い、などが挙げられる。特に手書き文字の認識では精度が低いと言われる。そこで手書き文字に対応したOCRを自作し、既存のものと比べどこまで精度が出せるかを試してみた。

## 対応文書
1.	スキャンされた手書き文字
2.	スキャンされた印刷文字   
      の2種類。現在**横書き**のみ対応している。
      
  
<img src="/images/ocr_images/スキャンされた手書き文字.jpg" width = "400">　　 <img src="/images/ocr_images/スキャンされた印刷文字.jpg" width = "400">  
　　　　　　　　　__スキャンされた手書き文字__　　　　　　　　　　　　　　　　　　　　__スキャンされた印刷文字__    
<br>  
<br> 
 
## OCRプログラム
**フォルダ名：/ocr/my_ocr.py**  
<br>
処理は大きく分けて①文字抽出、②文字予測の2つに分かれている。

### ①文字抽出
#### 1. 前処理  
<br>

**1)ファイル読み込み**

```
#テキスト化したい文書画像のパス名
filepath = "画像パス名"
img = cv2.imread(filepath)
gray_img = cv2.imread(filepath, 0)
data_form = int(input("文書の形式（スキャンされた手書きデータ:0  スキャンされた印刷データ:1）: "))
```
文書の形式のみ手動で選択する。  
<br>  
<br>  

**2)直線除去**  
```
#直線除去
gray_img = RemoveLinesFromImage(gray_img, "horizontal")
gray_img = RemoveLinesFromImage(gray_img, "vertical")
``` 
**RemoveLinesFromImage関数**により直線除去を行う。
処理内容はソーベルフィルタにより縦もしくは横方向にエッジ強調を行い、作成したエッジ強調画像に対しラベリング処理をかけ、ラベルの縦横比が15以上であれば線と判定し、除去を行う。この処理は縦方向・横方向に2回かけ、両方向の直線に対応させる。  
<br>

<img src="/images/ocr_images/直線除去_前.jpg" width = "300">  <img src="/images/ocr_images/直線除去_ソーベルフィルタ.jpg" width = "300">　 <img src="/images/ocr_images/直線除去_後.jpg" width = "300">  
　　　　　　　　__直線除去前__　　　　　　　　　__ソーベルフィルタ（上下方向）__　　　　　　　　　　__直線除去後__      
<br>
<br> 
  
**3)ノイズ除去**  
```
#ノイズ除去
if data_form == 0:
    denoised_img = RemoveNoiseFromImage(gray_img)
elif data_form == 1:
    _, denoised_img = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY)
```
ノイズ除去ではこの後行う文字抽出に向け、文字とそれ以外の背景との２値化を行うが、手書き文字と印刷文字で処理内容を変えている。理由として、印刷文字は文字のピクセル値がほぼ一定であるが、手書き文字では文字のなかでも色が濃い部分と薄い部分のムラがあり、単純な２値化では文字以外の除去が難しいためである。
手書き文字(data_form == 1)で利用する**RemoveNoiseFromImage関数**では2値化に平滑化処理を組み合わせている。  

<img src="/images/ocr_images/ノイズ除去_前.jpg" width = "400">  <img src="/images/ocr_images/ノイズ除去_後.jpg" width = "400">  
　　　　　　　　　　__ノイズ除去前__　　　　　　　　　　　　     　　　　__ノイズ除去後__  
<br>  
<br>
  
#### 2.行認識
文字列の行を認識する。   
<br>
  
**1)文字サイズの取得・図形の除去**  
```
#ラベリングによるレイアウト解析
binary_img = cv2.bitwise_not(denoised_img)
binary_img = cv2.erode(binary_img, kernel2)
binary_img = cv2.dilate(binary_img, kernel2)
label_numbers, labelimage, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(binary_img, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

#文字サイズの取得
char_area, char_height, char_width = GetCharSize(label_numbers, data)

#図形の除去
binary_img = RemoveFigure(binary_img, data, label_numbers, char_area)
``` 
**GetCharSize関数**により、ノイズ除去した画像に対しラベリング処理をかけ、得られた１つ１つのラベル情報(面積・高さ・幅)の中央値を平均的な文字の面積・高さ・幅の情報として取得する。その後、**RemoveFigure関数**を用いて得られた文字サイズの情報からラベルサイズが大きすぎるものは図形と判定し、除去を行う。
<br>  
<br>

**2)文字列の位置を認識**  
```
#文字列の位置取得
line_peaks = FindLinePeaks(binary_img, char_width, char_height, smooth_rate=4)

#文字同士の連結
binary_img = ConnectCharsOnLine(binary_img, line_peaks, labelimage, char_area)

#文字列の切り出し
binary_img, lines_position = DetectLines(binary_img)    
lines_position = sorted(lines_position.items())
```
**FindLinePeak関数**では文書の左右方向にピクセル値を合計したピクセル値カーブを作成し、そのピークの高さから行の位置を判定する。
**ConnectCharsOnLine関数**では判定した行の高さに対し、最も左側にある文字と最も右側にある文字をラベルの有無で判定する。文字かどうかはラベルの大きさがある一定を超えているかで判定する。その後、2文字間をつなぐ線を描画し、1つのその行全体が一つのラベルとして処理できるようにする。
**DetectLines関数**ではConnectCharsOnLine関数で結合した行全体に対し外接矩形を求め、矩形内が1行となるようにする。文字列全体が斜めになっている場合に対応するため、外接矩形は回転を考慮し、openCVのminAreaRect関数を用いる。複数の矩形が重なっている場合は矩形同士の重なった面積の割合から、合わせて1つの行と認識するか、異なる行と認識するかを判定する。  

<img src="/images/ocr_images/文字列の切り出し_1.jpg" width = "300">  <img src="/images/ocr_images/文字列の切り出し_2.jpg" width = "300">　 <img src="/images/ocr_images/文字列の切り出し_3.jpg" width = "300">  
　　　　　　　__行の位置判定__　　　　　　　　　　　　__行全体の結合__  　　　　 　　　     　　　　　　    __矩形の設置__
<br>  
<br>


**3)行の成形**   
```
#射影変換による行の成形
lines_list, line_image_list = ProcessingLines(lines_position, gray_img)

#行データを扱いやすい形に変更
transformed_lines_list = TransformLinesList(lines_list)
```
**ProcessingLines関数**では外接矩形の傾きを補正するため、矩形の高さ、幅はそのままで射影変換を行い、行全体を水平に変換する。
**TransformLinesList関数**ではそれぞれの行の左上のy軸ピクセル値を連番に変換し、この後の文字抽出で処理がしやすくなるようにする。  
<br>  
<br>

#### 3.文字の抽出  
文字列から1文字ずつ取り出す。  
<br>

**1)文字の位置を認識**  
```
#文字の検出
char_list = DetectChars(transformed_lines_list, line_image_list, data_form)
```
**DetectChars関数**では抽出したそれぞれの行に対し処理をかけ、さらに1文字ずつ抽出している。まず1つの行に対し上下方向にピクセル値を合計したピクセル値カーブを作成し、ピクセル合計値が0から1以上に変化している場所は文字の始まりの位置、1以上から０に変化している場所は文字の終わりの位置として認識する。その後文字の幅が狭すぎる場合は両サイドの文字との間や両サイドの文字の幅を確認し、1つの文字として結合するかを判定、結合を行う。また、文字の幅が広すぎる場合は複数の文字がつながっていると判定し、ピクセルカーブのピクセル合計値が小さな位置で分離を行い、文字同士を切り離す。文字結合や分離に用いたパラメータは文書の形式によって変えている。（規則正しく文字が並ぶ印刷文字と文字の大きさにばらつきがある手書き文字で同一のパラメータを用いるのは不適と考えたため。）最終的に抽出した1文字の枠内に対し、学習したAIモデルを適応するため、DetectChars関数内では**PreprocessingForLearning関数**を用いて文字予測の前処理を行っている。  
<br>
<img src="/images/ocr_images/抽出した文字(例).jpg" width = "600">  
　　　　　　　　　　　　　　　　__抽出した文字__  
             
<br>  
<br>

### ②文字予測
**1)モデルの読み込み**  
```
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
```
学習済みモデルのファイルと文字とラベル番号の対応関係を記録した.npyファイルを読み込む。（詳しくは学習の項を参照）
<br>  
<br>

**2)文字予測**  
```
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
```
DetectChars関数から得られた1文字1文字の画像データを学習済みモデルに渡し、文字予測を行い、テキストファイルとして書き起こす。行の高さが異なる場合は改行される。
<br>  
<br>

### 最終的な精度
__自作OCRプログラム__

<img src="/images/ocr_images/最終結果_手書き.jpg" width = "400">　<img src="/images/ocr_images/最終結果_印刷.jpg" width = "550">  
　　　　　　　　　__手書き文字__　　　　　　　　　　　    　　　　　　　 　　　　　　__印刷文字__  
<br>

最終的な精度は**80%**ほどの精度であった。やはり**ナ**と**十**や、 **C** と **(** など似た文字での認識ミスが多かった。このミスについてはAIでの予測後に文脈から文字を訂正する処理をかけることで対応可能だと考えられる。また、カタカナの濁音がデータベースになかったため、現状対応できていないことも大きい。文字の抽出過程でも１文字を２つに分けて抽出している場合もあり、さらなる改善の余地があると感じた。  
<br>
比較のため、既存のOCRツールとして**pyocr+tessaract**を用いて同じ画像に対し文字認識を行った結果を以下に示す。文字認識の前処理として2値化を行っている。  
<br>
__pyocr+tessaract__  
<img src="/images/ocr_images/tessaract_手書き文字.jpg" width = "400">　<img src="/images/ocr_images/tessaract_印刷文字.jpg" width = "550">   
　　　　　　　　　　__手書き文字__　　　　　　　　　　　　　　　　　　　　　     　　　　__印刷文字__ 
<br>  

pyocr+tessaractの結果では手書き文字で**30%**、印刷文字で**ほぼ100%** の精度であった。規則正しく並んだ文字に対しては既存のOCRツールのほうが自作OCRより精度が高かったが、手書き文字では大きく精度が落ちていた。罫線（直線）の除去等の処理を追加することで改善が見られそうだが、やはり手書き文字には手書き文字に特化したAIによる学習が有効だと感じた。
<br>
<br>

## 深層学習  
学習データにはETL1, ETL6, ETL9B, EMNISTの４つを組み合わせ、文字の被りがないように使用している。
データ内容は、  
<br>
**ETL1：カタカナ、特殊文字  
ETL6：特殊文字  
ETL9B：漢字(JIS第1水準)、ひらがな  
EMNIST-byclass：数字、アルファベット**  
<br>
であり、合計**3157文字**を分類した。各文字について**200枚**ずつ学習に使用している。
<br>  
<br>

## 深層学習の前段階  
### 文字データを前処理したのちpng画像として文字ごとにフォルダに保存するプログラム    
<br>

**フォルダ名  
ETL1：/training/ETL1_to_image.py    
ETL6：/training/ETL6_to_image.py    
ETL9B：/training/ETL9_to_image.py    
EMNIST-byclass：/training/EMNIST_to_image.py**     
<br>
前処理として、文字を画像の中心に持ってきて文字サイズが一定となるよう任意の大きさに拡大したのち、画像を64×64ピクセルにリサイズした。その後、png画像として作成したフォルダに保存している。この処理は各文字データベースごとに行っている。
<br>
<br>

文字データベース(ETL1等)　───　文字コード1　───　画像1  
　　　　　　　　　　　　　　　　　　　　　　 ───　画像2  
　　　　　　　　　　　　　　　　　　　　　　 ───　画像3  
　　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　 　　　───　画像200
<br>               
　　　　　　　　　　　　　───　文字コード2　───　画像1  
　　　　　　　　　　　　　　　　　　　 　　　  ───　画像2  
　　　　　　　　　　　　　　　　　　　　　  　　・  
　　　　　　　　　　　　　　　　　　　  　　　　・  
　　　　　　　　　　　　　　　　　　　  　　　　・  
                                          
　　　　　　　　__学習画像のディレクトリ構造__  
<br> 
<br>

### 学習データと検証データの画像パス名一覧を書き込んだテキストファイルと文字とラベル番号の対応関係を記録した.npyファイルを作成するプログラム
**フォルダ名：/training/OCR_make_text.py**  
<br>
学習時は画像量が多いため、ここで作成したテキストファイルから画像パス名をジェネレータにより適宜読み込み、メモリの節約を行う。  
<br>
<br>

## 深層学習プログラム
**フォルダ名：/training/OCR_torch.py**  
<br>

### パラメータ
学習はpytorchを使用し、**Efficientnet-b0を用いた転移学習**により行った。  
学習のパラメータは以下に示す。  
<br>
**バッチサイズ：512  
エポック数：30  
optimizer：Adam (学習率：1e-4)  
loss関数：CrossEntropyLoss  
GPU: NVIDIA TITAN RTX**  
<br>

学習に使用した画像は64×64のマトリクスサイズであり、転移学習によりImagenetで学習したパラメータを用いるためRGBの3チャンネルに変換する必要がある。各ラベル200枚の訓練データのうち、学習用データと検証用データを8:2の割合で分けて使用した。また、画像パスを読み込む際、ETL6から作成した画像のうち使用した文字は少なかったため、ETL1と同じファイルに入れてセットで読み込むようにした。  

<img src="/images/training_images/学習画像例_1.jpg" width = "250">　　<img src="/images/training_images/学習画像例_2.jpg" width = "250">　　<img src="/images/training_images/学習画像例_3.jpg" width = "210">  
　　　　　　　　　　　　　　　　　　　　　　　__学習画像例__  
<br>

### 水増し
```
transform1 = transforms.Compose([
                #回転・移動・縮小
                transforms.RandomRotation(degrees=5, fill=0),
                transforms.RandomAffine(degrees=[0, 0], translate=(0.1, 0.1), scale=(0.8, 1.2), fill=0),
                #Tensor型に変換
                transforms.ToTensor(),
                #色情報の標準化
                transforms.Normalize(0.5, 0.5, 0.5)
            ])

transform2 = transforms.Compose([
                #Tensor型に変換
                transforms.ToTensor(),
                #色情報の標準化
                transforms.Normalize(0.5, 0.5, 0.5)
            ])
```
学習時の水増しにはRandomRotationによる5度以内の回転、RandomAffineによる上下左右方向に10%以内の平行移動と0.8-1.2倍の拡大縮小処理を使用した。検証時には水増しを使用していない。    
<br>
<br>

### 学習結果  

```
EPOCHS:1 accuracy:23.9468 loss:0.0102 val_acc:75.8378 val_loss:0.0034  
EPOCHS:2 accuracy:83.9688 loss:0.0019 val_acc:94.7933 val_loss:0.0006  
EPOCHS:3 accuracy:93.6059 loss:0.0006 val_acc:96.6543 val_loss:0.0003  
EPOCHS:4 accuracy:95.7753 loss:0.0004 val_acc:97.2727 val_loss:0.0002  
EPOCHS:5 accuracy:96.7107 loss:0.0003 val_acc:97.6687 val_loss:0.0002  
EPOCHS:6 accuracy:97.2333 loss:0.0002 val_acc:97.9062 val_loss:0.0001  
EPOCHS:7 accuracy:97.6216 loss:0.0002 val_acc:97.9728 val_loss:0.0001  
EPOCHS:8 accuracy:97.8791 loss:0.0001 val_acc:98.1296 val_loss:0.0001  
EPOCHS:9 accuracy:98.0892 loss:0.0001 val_acc:98.2555 val_loss:0.0001  
EPOCHS:10 accuracy:98.2521 loss:0.0001 val_acc:98.2468 val_loss:0.0001  
EPOCHS:11 accuracy:98.3707 loss:0.0001 val_acc:98.2959 val_loss:0.0001  
EPOCHS:12 accuracy:98.4819 loss:0.0001 val_acc:98.3750 val_loss:0.0001  
EPOCHS:13 accuracy:98.5485 loss:0.0001 val_acc:98.3679 val_loss:0.0001  
EPOCHS:14 accuracy:98.6245 loss:0.0001 val_acc:98.4297 val_loss:0.0001  
EPOCHS:15 accuracy:98.6564 loss:0.0001 val_acc:98.5192 val_loss:0.0001  
EPOCHS:16 accuracy:98.7361 loss:0.0001 val_acc:98.3972 val_loss:0.0001  
EPOCHS:17 accuracy:98.7959 loss:0.0001 val_acc:98.4645 val_loss:0.0001  
EPOCHS:18 accuracy:98.8771 loss:0.0001 val_acc:98.4653 val_loss:0.0001  
EPOCHS:19 accuracy:98.8856 loss:0.0001 val_acc:98.4883 val_loss:0.0001  
EPOCHS:20 accuracy:98.9212 loss:0.0001 val_acc:98.4637 val_loss:0.0001  
EPOCHS:21 accuracy:98.9377 loss:0.0001 val_acc:98.5192 val_loss:0.0001  
EPOCHS:22 accuracy:98.9957 loss:0.0001 val_acc:98.4558 val_loss:0.0001   
EPOCHS:23 accuracy:99.0147 loss:0.0001 val_acc:98.5350 val_loss:0.0001  
EPOCHS:24 accuracy:99.0519 loss:0.0001 val_acc:98.5564 val_loss:0.0001  
EPOCHS:25 accuracy:99.0808 loss:0.0001 val_acc:98.4978 val_loss:0.0001  
EPOCHS:26 accuracy:99.1091 loss:0.0000 val_acc:98.5714 val_loss:0.0001  
EPOCHS:27 accuracy:99.1295 loss:0.0000 val_acc:98.5413 val_loss:0.0001  
EPOCHS:28 accuracy:99.1323 loss:0.0000 val_acc:98.5255 val_loss:0.0001  
EPOCHS:29 accuracy:99.1653 loss:0.0000 val_acc:98.5865 val_loss:0.0001  
EPOCHS:30 accuracy:99.1933 loss:0.0000 val_acc:98.5366 val_loss:0.0001  
```

　　　　　　　　　　　　　　　　　　　　　　__学習過程__   
                          
<br>

<img src="/images/training_images/学習曲線.jpg" width = "800">  

　　　　　　　　　　　　　　　　　　　　　　　__学習曲線__     
                          
<br>
<br>

最終的な学習のaccuracyは**98.5%** であった。20エポックあたりで精度が頭打ちになっていたため、early_stoppingの導入が有効と考えられる。また、Adam以外のoptimizerや学習率の変更などにより更なる精度の向上が期待できる。  
<br>
<br>  

## 参考文献  
**文字抽出**  
・[OpenCVを使って画像の射影変換をしてみるwithPython-Qlita](https://qiita.com/mix_dvd/items/5674f26af467098842f0)  
・[領域（輪郭）の特徴-OpenCV-Python Tutorials 1 documentation](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html)  
**深層学習**  
・[ETL文字データベース](http://etlcdb.db.aist.go.jp/?lang=ja)  
・[ETL文字データベース(etlcdb)を画像に変換する-Qlita](https://qiita.com/kcrt/items/a7f0582a91d6599d164d)  
・[EMNIST:手書きアルファベット&数字の画像データセット:AI・機械学習のデータセット辞典-@IT](https://atmarkit.itmedia.co.jp/ait/articles/2009/28/news024.html)  
・[転移学習で手書きのひらがな・漢字認識-Kludge Factory](https://tyfkda.github.io/blog/2022/05/26/hira-kan-recog.html)























 
 
 
