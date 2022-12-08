# 自作OCRプログラムの作成

## 目的
既存のOCRソフトは数多くあるが、日本語の文字認識は非常に難しいと言われている。理由として①ひらがな・カタカナに似た文字が多い、②「け」や「は」など独立した複数の画から構成される文字が多い、などが挙げられる。特に手書き文字の認識では精度が低いと言われる。そこで手書き文字に対応したOCRを自作し、既存のものと比べどこまで精度が出せるかを試してみた。

## 対応文書
1.	スキャンされた手書き文字
2.	スキャンされた印刷文字   
      の2種類。現状**横書き**のみ対応している。
      
  
<img src="/images/ocr_images/スキャンされた手書き文字.jpg" width = "500"> <img src="/images/ocr_images/スキャンされた印刷文字.jpg" width = "500">  
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

<img src="/images/ocr_images/ファイル読み込み.jpg" width = "700">  
文書の形式のみ手動で選択する。  
<br>  
<br>  
<br>

**2)直線除去**  
<img src="/images/ocr_images/直線除去.jpg" width = "500">  
**RemoveLinesFromImage関数**により直線除去を行う。
処理内容はソーベルフィルタにより縦もしくは横方向にエッジ強調を行い、作成したエッジ強調画像に対しラベリング処理をかけ、ラベルの縦横比が15以上であれば線と判定し、除去を行う。この処理は縦方向・横方向に2回かけ、両方向の直線に対応させる。  
<br>

<img src="/images/ocr_images/直線除去_前.jpg" width = "300">  <img src="/images/ocr_images/直線除去_ソーベルフィルタ.jpg" width = "300">　 <img src="/images/ocr_images/直線除去_後.jpg" width = "300">  
　　　　　　　　__直線除去前__　　　　　　　　　　__ソーベルフィルタ（上下方向）__　　　　　　　　　　__直線除去後__      
<br>
<br> 
  
**3)ノイズ除去**  
<img src="/images/ocr_images/ノイズ除去.jpg" width = "400">  
この後行う文字抽出に向け、**RemoveNoiseFromImage関数**では2値化等のノイズ除去処理を行う。  

<img src="/images/ocr_images/ノイズ除去_前.jpg" width = "400">  <img src="/images/ocr_images/ノイズ除去_後.jpg" width = "400">  
　　　　　　　　　　__ノイズ除去前__　　　　　　　　　　　　     　　　　__ノイズ除去後__  
<br>  
<br>
  
#### 2.行認識
文字列の行を認識する。   
<br>
  
**1)文字サイズの取得・図形の除去**  
<img src="/images/ocr_images/文字サイズの取得・図形の除去.jpg" width = "900">  
**GetCharSize関数**により、ノイズ除去した画像に対しラベリング処理をかけ、得られた１つ１つのラベル情報(面積・高さ・幅)の中央値を平均的な文字の面積・高さ・幅の情報として取得する。その後、**RemoveFigure関数**を用いて得られた文字サイズの情報からラベルサイズが大きすぎるものは図形と判定し、除去を行う。
<br>  
<br>

**2)文字列の位置を認識**  
<img src="/images/ocr_images/文字列の切り出し.jpg" width = "00">  
**FindLinePeak関数**では文書の左右方向にピクセル値を合計したピクセル値カーブを作成し、そのピークの高さから行の位置を判定する。
**ConnectCharsOnLine関数**では判定した行の高さに対し、最も左側にある文字と最も右側にある文字をラベルの有無で判定する。文字かどうかはラベルの大きさがある一定を超えているかで判定する。その後、2文字間をつなぐ線を描画し、1つのその行全体が一つのラベルとして処理できるようにする。
**DetectLines関数**ではConnectCharsOnLine関数で結合した行全体に対し外接矩形を求め、矩形内が1行となるようにする。文字列全体が斜めになっている場合に対応するため、外接矩形は回転を考慮し、openCVのminAreaRect関数を用いる。複数の矩形が重なっている場合は矩形同士の重なった面積の割合から、合わせて1つの行と認識するか、異なる行と認識するかを判定する。  

<img src="/images/ocr_images/文字列の切り出し_1.jpg" width = "300">  <img src="/images/ocr_images/文字列の切り出し_2.jpg" width = "300">　 <img src="/images/ocr_images/文字列の切り出し_3.jpg" width = "300">  
　　　　　　　__行の位置判定__　　　　　　　　　　　　__行全体の結合__  　　　　 　　　     　　　　　　    __矩形の設置__
<br>  
<br>


**3)行の成形**   
<img src="/images/ocr_images/行の成形.jpg" width = "600">  
**ProcessingLines関数**では外接矩形の傾きを補正するため、矩形の高さ、幅はそのままで射影変換を行い、行全体を水平に変換する。
**TransformLinesList関数**ではそれぞれの行の左上のy軸ピクセル値を連番に変換し、この後の文字抽出で処理がしやすくなるようにする。  
<br>  
<br>

#### 3.文字の抽出  
文字列から1文字ずつ取り出す。  
<br>

**1)文字の位置を認識**  
<img src="/images/ocr_images/文字の抽出.jpg" width = "600">  
**DetectChars関数**では抽出したそれぞれの行に対し処理をかけ、さらに1文字ずつ抽出している。まず1つの行に対し上下方向にピクセル値を合計したピクセル値カーブを作成し、ピクセル合計値が0から1以上に変化している場所は文字の始まりの位置、1以上から０に変化している場所は文字の終わりの位置として認識する。その後文字の幅が狭すぎる場合は両サイドの文字との間や両サイドの文字の幅を確認し、1つの文字として結合するかを判定、結合を行う。また、文字の幅が広すぎる場合は複数の文字がつながっていると判定し、ピクセルカーブのピクセル合計値が小さな位置で分離を行い、文字同士を切り離す。文字結合や分離に用いたパラメータは文書の形式によって変えている。（規則正しく文字が並ぶ印刷文字と文字の大きさにばらつきがある手書き文字で同一のパラメータを用いるのは不適と考えたため。）最終的に抽出した1文字の枠内に対し、学習したAIモデルを適応するため、DetectChars関数内では**PreprocessingForLearning関数**を用いて文字予測の前処理を行っている。  
<br>
<img src="/images/ocr_images/抽出した文字(例).jpg" width = "600">  
　　　　　　　　　　　　　　　　__抽出した文字__  
             
<br>  
<br>

### ②文字予測
**1)モデルの読み込み**  
<img src="/images/ocr_images/モデルの読み込み.jpg" width = "600">  
学習済みモデルのファイルと文字とラベル番号の対応関係を記録した.npyファイルを読み込む。（詳しくは学習の項を参照）
<br>  
<br>

**2)文字予測**  
<img src="/images/ocr_images/文字予測.jpg" width = "450">   
DetectChars関数から得られた1文字1文字の画像データを学習済みモデルに渡し、文字予測を行い、テキストファイルとして書き起こす。行の高さが異なる場合は改行される。
<br>  
<br>

### 最終的な精度
<img src="/images/ocr_images/最終結果_手書き.jpg" width = "400">　<img src="/images/ocr_images/最終結果_印刷.jpg" width = "550">  
　　　　　　　__テキスト化後（手書き文字）__　　　　　　　　　　　     　　　　__テキスト化後（印刷文字）__  
<br>

最終的な精度は**8割**ほどの精度であった。やはり**ナ**と**十**や、 **C** と **(** など似た文字での認識ミスが多かった。このミスについてはAIでの予測後に文脈から文字を訂正する処理をかけることで対応可能だと考えられる。また、カタカナの濁音がデータベースになかったため、現状対応できていないことも大きい。文字の抽出過程でも１文字を２つに分けて抽出している場合もあり、さらなる改善の余地があると感じた。
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

文字データベース(ETL1等)　‐---　文字コード1　------　画像1  
　　　　　　　　　　　　　　　　　　　　　------　画像2  
　　　　　　　　　　　　　　　　　　　　　------　画像3  
　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　------　画像200
<br>               
　　　　　　　　　　　　----　文字コード2　------　画像1  
　　　　　　　　　　　　　　　　　　　　　------　画像2  
　　　　　　　　　　　　　　　　　　　　　　・  
　　　　　　　　　　　　　　　　　　　　　　・                                          
　　　　　　　　　　　　　　　　　　　　　　・   
<br>                   
　　　　　　　　__学習画像のディレクトリ構造__  
<br> 
<br>

### 学習データと検証データの画像パス名一覧を書き込んだテキストファイルと文字とラベル番号の対応関係を記録した.npyファイルを作成するプログラム
**フォルダ名：/training/OCR_make_text.py**  
<br>
学習時は画像量が多いため、ここで作成したテキストファイルから画像パス名をジェネレータにより適宜読み込み、メモリの節約を行う。  

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
                       

### 水増し
<img src="/images/training_images/水増し.jpg" width = "800">  
学習時の水増しにはRandomRotationによる5度以内の回転、RandomAffineによる上下左右方向に10%以内の平行移動と0.8-1.2倍の拡大縮小処理を使用した。検証時には水増しを使用していない。  
<br>

### 学習結果  
<br>

<img src="/images/training_images/学習過程.jpg" width = "800">    
　　　　　　　　　　　__学習過程__    
<br>

<img src="/images/training_images/学習曲線.jpg" width = "800">  
　　　　　　　　　　　　　__学習曲線__   
<br>

最終的な学習のaccuracyは98.5%であった。20エポックあたりで精度が頭打ちになっていたため、early_stoppingの導入が有効と考えられる。また、Adam以外のオプティマイザーや学習率の変更などにより更なる精度の向上が期待できる。




















 
 
 
