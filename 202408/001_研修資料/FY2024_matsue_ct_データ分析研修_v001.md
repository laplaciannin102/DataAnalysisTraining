
# データ分析研修
## memo

- 最終更新日：2024/08/31

---

## 対象読者・前提知識
### 対象読者

- データサイエンティストとして、データ分析業務に従事する人

### 前提知識

- WindowsのPCの基本的な使い方を理解していること

---

## セットアップ
### テキストエディタ

- 基本的に好きなものを用いて構いません
  - 特に好みが無い場合は、VSCode(Visual Studio Code)を推奨します
  - その他ソフト例
    - Atom
    - Vim
    - Sublime Text
    - サクラエディタ

1. 次のページを参考にしながら、VSCodeをインストールしてください。
   - [Windows への Visual Studio Code のインストール方法](https://www602.math.ryukoku.ac.jp/Prog1/vscode-win.html)

#### 参考

- [Visual Studio Code の Download ページ](https://code.visualstudio.com/download)


### Python

- データ分析ではPythonというプログラミング言語を用いることが多いです。

1. 次のページを参考にしながら、Pythonをインストールしてください。
   - [Windows版Anacondaのインストール](https://www.python.jp/install/anaconda/windows/install.html)

#### 参考

- [Python環境構築](https://github.com/laplaciannin102/jissen-marketing/blob/master/001_%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89/001_%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89.md)


### R

- データ分析ではRというプログラミング言語を用いることが多いです。
  - R言語を扱う場合には、統合開発環境であるRStudioを用いることが多いです。

1. 次のページを参考にしながら、RとRStudioをインストールしてください。
   - [初心者向けRのインストールガイド](https://syunsuke.github.io/r_install_guide_for_beginners/)
     - [Rのインストール](https://syunsuke.github.io/r_install_guide_for_beginners/03_installation_of_R.html)
     - [RStudioのインストール](https://syunsuke.github.io/r_install_guide_for_beginners/04_installation_of_RStudio.html)


### Git

- Gitは、プログラムのソースコードなどの変更履歴を記録・追跡するための(分散型の)バージョン管理システムです。

1. 次のページの[入門編](https://backlog.com/ja/git-tutorial/intro/01/)をクリックし、指示に従いながら順に読み進めてください。
   - [サル先生のGit入門](https://backlog.com/ja/git-tutorial/)

2. 1を読み進めていけば見つかりますが、Gitのインストール方法は次のページになります。
   - [Gitのインストール](https://backlog.com/ja/git-tutorial/intro/05/)
   - 「お使いのパソコンの環境を選んでください。」という項目はプルダウンで「Windowsを選択」を選択してください。

3. 研修では入門編の「チュートリアル1 Gitの基本」まで読み進め、「リポジトリの共有」以降は空き時間、または実際に業務でGitを用いるタイミングで読んでください。

#### 参考

- [[ブレインパッド]Gitハンズオン研修 / Git Hands-on](https://speakerdeck.com/brainpadpr/git-hands-on)
- [GitHub演習](https://github.com/kaityo256/github)


---

## プログラミング基礎
### Pythonの基本

1. 次のページを読み進めながら、実際のPythonの使い方を学びましょう。
   - [ゼロからのPython入門講座](https://www.python.jp/train/index.html)

- ※必要な項目のみ読めば大丈夫です

#### 参考

- [[東京大学]Pythonプログラミング入門](https://utokyo-ipp.github.io/)
  - 東大のPython教材
- [Python早見帳](https://chokkan.github.io/python/index.html)
  - Python早見帳は、Pythonのプログラムと実行例をさっと確認（早見）できるJupyter Notebook（帳）です。

---

## 数学・統計学
### 基礎統計学

- データ分析において、数学や統計学についての基本的な理解をしておくことは重要です。

1. 次のスライドを読み進め、基礎統計学について学びましょう。
  - [[ブレインパッド]【新卒研修資料】基礎統計学 / Basic of statistics](https://speakerdeck.com/brainpadpr/basic-of-statistics)

- ※データサイエンティストとして分析業務に従事する際は、統計検定2級程度の知識があると良いとされています

#### 参考

- [[とけたろう]【完全網羅】統計検定２級チートシート](https://toketarou.com/cheatsheet/)
- [[Qiita]統計検定2級チートシート](https://qiita.com/moriokumura/items/ed246efe11fc5d3fc760)

---

## 機械学習
### 機械学習

- データ分析において、機械学習についての基本的な理解をしておくことは重要です。

1. 次のスライドを読み進め、機械学習について学びましょう。
   - [[ブレインパッド]分析の基礎（モデリング）/ Basics of analysis ~modeling~](https://speakerdeck.com/brainpadpr/basics-of-analysis-modeling)

   - 目次:
     - モデリングの考え方
     - 確率分布
     - 教師あり学習
     - モデルの精度を上げるための工夫
     - 教師なし学習

#### 参考

- [機械学習帳](https://chokkan.github.io/mlnote/index.html)
  - 機械学習帳は、機械学習を学ぶためのノート（帳）を、デジタル（機械）による新しいカタチの学習帳として実現することを目指しています。

---

## その他分析に必要な知識
### 探索的データ分析

- データの分析を行う最初の段階で、どのようなデータが与えられているか探索的に確認する作業を行い、この作業を**探索的データ分析(EDA)**と呼びます。

1. 次のページを読み進め、探索的データ分析について学びましょう。
   - [【データサイエンティスト入門編】探索的データ解析（EDA）の基礎操作をPythonを使ってやってみよう](https://www.codexa.net/basic-exploratory-data-analysis-with-python/)

#### 参考

- [探索的データ分析](https://uribo.github.io/practical-ds/01/eda.html)
  - R言語を用いた場合

### 可視化

- データ分析を行う際に、可視化(Visualization)は重要です。
- Pythonを用いて可視化を行う場合、**matplotlib**や**seaborn**などのライブラリがよく用いられます。

1. 次のページを読み進め、matplotlibを用いた可視化について学びましょう。
   - [matplotlibを利用したデータ可視化（data visualization）の基礎](https://www.hello-statisticians.com/python/data-vis-matplotlib-01.html)

2. 次のページを読み進め、seabornを用いた可視化について学びましょう。
   - [seabornを利用したデータ可視化（data visualization）の基礎](https://www.hello-statisticians.com/python/data-vis-seaborn-01.html)

#### 参考

- [[東京大学]簡単なデータの可視化](https://utokyo-ipp.github.io/appendix/3-visualization.html)


### スクレイピング

- スクレイピングとは、Webサイトからデータを収集するための技術です。

1. 次のページを読み進め、スクレイピングとは何か、またPythonによるスクレイピングの方法をざっと学びましょう。
   - [【初心者向け・保存版】PythonでWebスクレイピングしてみよう！](https://aiacademy.jp/media/?p=2116)


2. 次のページも読み進め、理解を深めましょう。
   - [[Qiita]Python Webスクレイピング 実践入門](https://qiita.com/Azunyan1111/items/9b3d16428d2bcc7c9406)

3. 2の記事にも記載がありますが、スクレイピングには**注意するべき点**があります。次の2つのページをよく読んでおいてください。
   - [[Qiita]Webスクレイピングの注意事項一覧](https://qiita.com/nezuq/items/c5e827e1827e7cb29011)
   - [岡崎市立中央図書館事件(Librahack事件) - Wikipedia](https://ja.wikipedia.org/wiki/%E5%B2%A1%E5%B4%8E%E5%B8%82%E7%AB%8B%E4%B8%AD%E5%A4%AE%E5%9B%B3%E6%9B%B8%E9%A4%A8%E4%BA%8B%E4%BB%B6)

---

## Appendix

### ディープラーニング

- [Chainer Tutorial](https://tutorials.chainer.org/ja/tutorial.html)
  - Chainerの開発をしていたPFN(Preferred Networks)の記事
  - ディープラーニングなどについて詳しい解説

- [Deep Learning入門](https://www.youtube.com/playlist?list=PLg1wtJlhfh23pjdFv4p8kOBYyTRvzseZ3)
  - SONYの人がDeep Learningについてざっと解説

### マーケティング

- no contents

### 参考URL

- [OpenBrainPad Project](https://brainpad.github.io/OpenBrainPad/)

- [OR事典Wiki](https://orsj-ml.org/orwiki/wiki/index.php?title=%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)
  - 最適化などについて様々な記事があります

- [数理・データサイエンス教育強化拠点コンソーシアム](mi.u-tokyo.ac.jp/consortium/index.html)
  - [YouTubeチャンネル](https://www.youtube.com/@user-xj8wh7tt5b)

- [AIcia Solid Project](https://www.youtube.com/@AIcia_Solid)
  - データサイエンスVTuber
  - かなり詳しいです
  - 中の人は、「アトラエ(転職サイトGreenとかの会社)」のデータサイエンティスト「杉山聡」さん

### 参考図書紹介

- [統計的学習の基礎](https://www.kyoritsu-pub.co.jp/book/b10004471.html)

- [[TJO]2023年版：実務データ分析を手掛けるデータサイエンティスト向け推薦書籍リスト（初級6冊＋中級8冊＋テーマ別15冊）](https://tjo.hatenablog.com/entry/2023/02/07/170000)

- [[TJO]2022年版：実務の現場で働くデータサイエンティスト向け推薦書籍リスト（初級5冊＋中級8冊＋テーマ別14冊）](https://tjo.hatenablog.com/entry/2022/02/09/170000)





