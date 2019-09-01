# データ分析研修
## 日時
- 第1回_2019/08/25_13:00~17:00
- 第2回_2019/09/01_13:00~17:00

## はじめに
- 全般
  - 全部の資料を用意するのは大変なので、基本的にはQiitaの入門記事などを参照していきます
- テキスト共有ページ(このページのことです。このURLにアクセスしてください。)
    - https://mimemo.io/m/5dn7jlKVKjor9Ye

## 自習課題
- [Progate] できればPython研修の3まで
  - https://prog-8.com/

## Python
- Pythonについて全般
  - Python-izm
    - 基礎編と入門編だけでも見ておくと良いと思います。
    - https://www.python-izm.com/introduction/

- インストール
  - Pythonは3系(今だと3.7?)を使う
  - Anaconda
    - Anaconda はデータサイエンス向けのPythonパッケージなどを提供するプラットフォームです。科学技術計算などを中心とした、多くのモジュールやツールのコンパイル済みバイナリファイルを提供しており、簡単にPythonを利用する環境を構築できます。
    - https://www.anaconda.com/distribution/
  - Graphviz
    - Python にかぎらず、グラフ（決定木、ノードとエッジの集合、グラフ理論のグラフ）を記述するときに利用するツールパッケージ
    - http://ruby.kyoto-wu.ac.jp/info-com/Softwares/Graphviz/
      - path部分は「C:\Program Files\Anaconda\Library\bin\graphviz」とか？で良い

  - その他ライブラリインストール
    - matplotlib日本語化
    ```
    pip install japanize-matplotlib
    ```

    - 決定木可視化(hoge viz無しで)
    ```
    pip install dtreeplt
    ```

- Jupyter Notebookについて
  - コマンドライン上で「jupyter notebook」を実行
  - https://qiita.com/horankey_jet_city/items/f29c3477a5099f12cb18
  - こんなバッチファイルを用意しておくとDドライブ上で起動できたりして便利だったりする
    - start_jupyter.bat
    ```
    cd /d %~dp0
    jupyter notebook
    ```
    - KernelタブのRestart & Clear Output でノート全体をプログラム実行前に戻せる

- Pandas基本操作
  - https://qiita.com/ysdyt/items/9ccca82fc5b504e7913a

## 機械学習
- 機械学習について
  - https://qiita.com/taki_tflare/items/42a40119d3d8e622edd2

## データ分析
- Pythonでタイタニック生存予測
  - https://www.randpy.tokyo/entry/python_random_forest

## クローリング、スクレイピング
- PythonとBeautiful Soupでスクレイピング
  - https://qiita.com/itkr/items/513318a9b5b92bd56185
- Python Webスクレイピング 実践入門
  - https://qiita.com/Azunyan1111/items/9b3d16428d2bcc7c9406
- PythonでWeb上の画像などのファイルをダウンロード（個別・一括）
  - https://note.nkmk.me/python-download-web-images/
- 【Python】Googleの検索結果をアクセス制限なしで取得する
  - https://qiita.com/derodero24/items/949ac666b18d567e9b61

# プログラムとか共有する用
```
import numpy as np
import pandas as pd
```
```
df = pd.DataFrame([[160,50,20],[170,60,26],[180,70,30]], columns=['身長', '体重', '年齢'])
df
```
```
for ii in range(3):
    print('iiは ', ii)
    
    for jj in range(5):
        print('ii:', ii)
        print('jj:', jj)
        print()
        print('■■'*5)
```


