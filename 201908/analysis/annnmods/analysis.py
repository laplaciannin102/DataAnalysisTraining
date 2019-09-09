#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
分析用module
"""


import sys, os
import gc
import time
import numpy as np
import pandas as pd
import codecs
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# 評価関数
# 分類
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# 回帰
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 設定のimport
from .mod_config import *
# 自作moduleのimport
from .calculation import *
from .useful import *
from .scraping import *
from .visualization import *
from .preprocessing import *


def pro_read_csv(path, encoding='utf-8', usecols=None):
    """
    pd.read_csv()で読めないcsvファイルを読み込む。

    Args:
        path: str
            読み込むファイルのパス。

        encoding: str, optional(default='utf-8)
            エンコード。

        usecols: list of str, optional(default=None)
            指定したカラムのみを読み込む場合に使用。

    Returns:
        df: DataFrame
            読み込んだDataFrame。
    """
    if usecols is None:
        print('usecols: all columns')
        with codecs.open(path, 'r', encoding, 'ignore') as file:
            df = pd.read_table(file, delimiter=',')
    else:
        # 指定したカラムのみ読み込む場合
        print('usecols:', usecols)
        with codecs.open(path, 'r', encoding, 'ignore') as file:
            df = pd.read_table(file, delimiter=',', usecols=usecols)

    return df


def check_var_info(df, target=None, file_name='', filecol_is=False, transcol_is=True):
    """
    DataFrameの各カラムの示す変数の概要をまとめる。

    Args:
        df: DataFrame
            概要を調べるDataFrame。
        
        target: str, optional(default=None)
            目的変数となるカラム名。Noneの時はtarget関連のカラムは作成しない。
        
        file_name: str, optional(default='')
            DataFrameが保存されていたファイルの名前。一応パスではない想定。
        
        filecol_is: bool, optional(default=False)
            file_nameカラムを作成するかどうか。Trueで作成する。
        
        transcol_is: bool, optional(default=True)
            カラム名の日本語訳を行うかどうか。Trueで行う。

    Returns:
        var_info_df: DataFrame
            各カラムの示す変数の概要をまとめたDataFrame。
            columns:
                file_name:          ファイル名
                var_name:           変数名
                var_name_ja:        変数名の日本語訳
                dtype:              データ型
                n_unique:           値の種類数
                mode:               最頻値
                count_of_mode:      最頻値が占めるレコード数
                missing_rate:       欠損率
                n_exist:            非欠損数
                n_missing:          欠損数
                n_rows:             行数。レコード数。
                mean:               平均値
                std:                標準偏差
                min:                最小値
                med:                中央値
                max:                最大値
                corr_with_target:   目的変数との相関
                abs_corr_with_target: 目的変数との相関の絶対値
    
    Examples:
    
    # タイタニック生存予測
    df = sns.load_dataset('titanic')
    var_info_df = check_var_info(df)
    # var_info_dfのカラム名を日本語にしたい場合
    enja_dict = {
        'file_name': 'ファイル名',
        'var_name': '変数名',
        'var_name_ja': '変数名日本語訳',
        'dtype': 'データ型',
        'n_unique': '値の種類数',
        'mode': '最頻値',
        'count_of_mode': '最頻値レコード数',
        'missing_rate': '欠損率',
        'n_exist': '非欠損数',
        'n_missing': '欠損数',
        'n_rows': 'レコード数',
        'mean': '平均値',
        'std': '標準偏差',
        'min': '最小値',
        'med': '中央値',
        'max': '最大値',
        'corr_with_target': '目的変数との相関',
        'abs_corr_with_target': '目的変数との相関_絶対値'
    }
    var_info_df = var_info_df.rename(columns=enja_dict)
    """
    # 各変数についてまとめたい事柄(今後増やしていきたい)
    var_info_cols = [
        'file_name',        # ファイル名
        'var_name',         # 変数名
        'var_name_ja',      # 変数名の日本語訳
        'dtype',            # データ型
        'n_unique',         # 値の種類数
        'mode',             # 最頻値
        'count_of_mode',    # 最頻値が占めるレコード数
        'missing_rate',     # 欠損率
        'n_exist',          # 非欠損数
        'n_missing',        # 欠損数
        'n_rows'            # 行数。レコード数。
    ]

    key = 'var_name'
    var_info_df = pd.DataFrame(columns=[key])
    df_cols = df.columns.tolist()
    var_info_df[key] = df_cols

    # 基礎統計料
    basic_stat_cols = ['mean', 'std', 'min', 'med','max']
    var_info_cols += basic_stat_cols

    # targetとの相関
    # targetがNoneでなく、df_colsの中にある時
    if (not target is None) and (target in df_cols):
        corr_cols = ['corr_with_target', 'abs_corr_with_target']
        var_info_cols += corr_cols

    if not filecol_is:
        var_info_cols.remove('file_name')
    
    if not transcol_is:
        var_info_cols.remove('var_name_ja')
    
    # ファイル名
    if filecol_is:
        var_info_df['file_name'] = file_name

    # 日本語訳
    if transcol_is:
        try:
            var_info_df['var_name_ja'] = var_info_df[key].apply(translate_to_ja)
            time.sleep(0.4)
        except:
            print('google translate error')
            var_info_df['var_name_ja'] = 'translate error'

    # データ型
    dtype_df = pd.DataFrame(df.dtypes).reset_index()
    dtype_df.columns = [key, 'dtype']
    dtype_df['dtype'] = dtype_df['dtype'].astype(str)
    var_info_df = pd.merge(var_info_df, dtype_df, on=key, how='inner')

    # 値の種類数
    nunique_df = pd.DataFrame(df.nunique()).reset_index()
    nunique_df.columns = [key, 'n_unique']
    nunique_df['n_unique'] = nunique_df['n_unique'].astype(int)
    var_info_df = pd.merge(var_info_df, nunique_df, on=key, how='inner')

    # 最も多数派の値
    major_list = []
    for var_name in df_cols:
        vc_df = pd.DataFrame(df[var_name].value_counts())
        cmx = vc_df[var_name].max()
        if cmx >= 2:
            vc_df = vc_df[vc_df[var_name]==cmx]
            major_vals = vc_df.index.tolist()
            major_vals_text = ''
            for val in major_vals:
                major_vals_text += str(val) + '/'
            major_vals_text = major_vals_text[:-1]
        else:
            # 1の時は多数派の概念を考えない
            major_vals_text = ''
        major_list += [[var_name, major_vals_text, cmx]]
    major_df = pd.DataFrame(major_list, columns=[key, 'mode', 'count_of_mode'])
    var_info_df = pd.merge(var_info_df, major_df, on=key, how='inner')
    

    # 非欠損数
    n_exist_df = pd.DataFrame(df.notnull().sum()).reset_index()
    n_exist_df.columns = [key, 'n_exist']
    var_info_df = pd.merge(var_info_df, n_exist_df, on=key, how='inner')

    # 欠損数
    n_missing_df = pd.DataFrame(df.isnull().sum()).reset_index()
    n_missing_df.columns = [key, 'n_missing']
    var_info_df = pd.merge(var_info_df, n_missing_df, on=key, how='inner')

    # 行数
    var_info_df['n_rows'] = len(df)

    # 欠損率
    e = 'n_exist'
    m = 'n_missing'
    var_info_df['missing_rate'] = var_info_df.apply(lambda row: row[m]/(row[e]+row[m]) if ((row[e]+row[m])!=0) else 0, axis=1)

    # 基礎統計量
    bst_df = df.describe()
    bst_df = bst_df.T.rename(columns={'50%': 'med'})[basic_stat_cols].reset_index(drop=False)
    bst_df.columns = [key] + basic_stat_cols
    var_info_df = pd.merge(var_info_df, bst_df, on=key, how='outer')

    # targetとの相関
    # targetがNoneでなく、df_colsの中にある時
    if (not target is None) and (target in df_cols):
        corr_df = df.corr()
        target_corr_df = corr_df[[target]]
        target_corr_df[corr_cols[1]] = target_corr_df[target].apply(abs)
        target_corr_df = target_corr_df.reset_index()
        target_corr_df.columns = [key, corr_cols[0], corr_cols[1]]
        var_info_df = pd.merge(var_info_df, target_corr_df, on=key, how='outer')
        del corr_df
        gc.collect()


    # 整理
    var_info_df = var_info_df.reset_index(drop=True)[var_info_cols]

    return var_info_df


def check_data_list(path_list, encoding='utf-8'):
    """
    各ファイルの概要をまとめたDataFrameを返す。
    csvとxlsxに対応。

    Args:
        path_list: list of str
            概要を知りたいファイルパスのリスト。

        encoding: str, optional(default='utf-8)
            各ファイルを読み込むときのエンコード。

    Returns:
        data_info_df, var_info_df: tuple
            data_info_df: DataFrame
                データの概要をまとめたDataFrame。

            var_info_df: DataFrame
                カラムの概要をまとめたDataFrame。
                check_var_info()参照。
    """
    n_files = len(path_list)    # file数
    print('num of all files:', n_files)

    # dataの概要をまとめたDataFrame
    info_cols = ['path', 'file_name', 'n_rows', 'n_cols', 'columns']
    data_info_df = pd.DataFrame(columns=info_cols)

    # 各カラムの概要をまとめたDataFrame
    var_info_cols = [
        'file_name',        # ファイル名
        'var_name',         # 変数名
        'var_name_ja',      # 変数名の日本語訳
        'dtype',            # データ型
        'n_unique',         # 値の種類数
        'mode',             # 最頻値
        'count_of_mode',    # 最頻値が占めるレコード数
        'missing_rate',     # 欠損率
        'n_exist',          # 非欠損数
        'n_missing',        # 欠損数
        'n_rows'            # 行数。レコード数。
    ]
    # 基礎統計料
    basic_stat_cols = ['mean', 'std', 'min', 'med','max']
    var_info_cols += basic_stat_cols
    var_info_df = pd.DataFrame(columns=var_info_cols)

    # 各ファイルについてまとめる
    for path in path_list:
        path = str(path)
        file_name = path.split('/')[-1].split('\\')[-1]     # ファイル名
        extension = file_name.split('.')[-1]     # 拡張子

        if extension == 'csv':
            df = pro_read_csv(path, encoding=encoding)
        elif extension == 'xlsx':
            df = pd.read_excel(path, index=False)
        else:
            # csvでもxlsxでも無ければ一旦無視する。
            continue

        shape = df.shape
        n_rows = int(shape[0])   # 行数。レコード数。
        n_cols = int(shape[1])   # 列数。カラム数。
        columns = df.columns.tolist()   # カラム一覧。

        tmp = pd.DataFrame([[path, file_name, n_rows, n_cols, columns]], columns=info_cols)
        data_info_df = data_info_df.append(tmp)

        var_tmp = check_var_info(df, target=None, file_name=file_name, filecol_is=True, transcol_is=True)
        var_info_df = var_info_df.append(var_tmp)

        # 掃除
        del df, tmp, var_tmp
        gc.collect()

    data_info_df = data_info_df.reset_index(drop=True)
    var_info_df = var_info_df.reset_index(drop=True)
    print('num of files of data:', len(data_info_df))

    # 戻り値
    ret = data_info_df, var_info_df
    return ret


def print_clf_score(true_y, pred_y):
    """
    分類モデルの評価関数の値を表示する。
    sklearn.metrics.classification_report も参考に。

    Args:
        true_y: 1d array-like, or label indicator array / sparse matrix
            実測値。

        pred_y: 1d array-like, or label indicator array / sparse matrix
            予測値。
    
    Returns:
        score_df, cm_df: tuple
            score_df: DataFrame
                分類問題の評価関数の値をまとめたDataFrame。
                columns: Accuracy, F-measure, Precision, Recall, Log_Loss, ROC_AUC
            
            cm_df: DataFrame
                混同行列のDataFrame。
    
    Remarks:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    acc = accuracy_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    l_loss = log_loss(true_y, pred_y)
    roc_auc = roc_auc_score(true_y, pred_y)

    score_df = pd.DataFrame(
        [[acc, f1, precision, recall, l_loss, roc_auc]],
        columns=['Accuracy', 'F-measure', 'Precision', 'Recall', 'Log_Loss', 'ROC_AUC']
        )

    # 混同行列
    cm_df = pd.DataFrame(confusion_matrix(pred_y, true_y))
    cm_df.columns = ['Negative', 'Positive']
    cm_df.index = ['True', 'False']

    print('Accuracy:', pro_round(acc, 2))
    print('f1-measure:', pro_round(f1, 2))
    print('Precision:', pro_round(precision,2))
    print('Recall:', pro_round(recall, 2))
    print('Log Loss:', pro_round(l_loss, 2))
    print('ROC AUC:', pro_round(roc_auc, 2))
    
    return score_df, cm_df


def print_reg_score(true_y, pred_y):
    """
    回帰モデルの評価関数の値を表示する。

    Args:
        true_y: 1d array-like, or label indicator array / sparse matrix
            実測値。

        pred_y: 1d array-like, or label indicator array / sparse matrix
            予測値。
    
    Returns:
        score_df: DataFrame
            回帰問題の評価関数の値をまとめたDataFrame。
            columns: MSE, R2_score
    
    Remarks:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    mse = mean_squared_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)

    score_df = pd.DataFrame(
        [[mse, r2]],
        columns=['MSE', 'R2_score']
        )

    print('mean squared error:', pro_round(mse, 2))
    print('R^2:', pro_round(r2, 2))
    
    return score_df


def extract_isone(row, cols=None, sep='/'):
    """
    辞書型のrowの指定された各カラムのうち1かTrueであるものを抜き出す。
    ダミー変数化した後のDataFrameなどに使用。
    DataFrameに対してapply()の中で使用する想定。
    使用例は下記Examples参照。

    Args:
        row: dict
            DataFrameの各行を表すdict。
        
        cols: list of str or tuple of str, optional(default=None)
            判定したいカラム名のリスト。
            初期設定のNoneが選ばれた場合、rowのkey全てが選ばれる。
        
        sep: str, optional(default='/')
            出力で使用する文字列の区切り文字。
    
    Returns:
        isone_list, isone_text: tuple
            isone_list: list of str
                値が1かTrueであるカラム名のリスト。
            
            isone_text: str
                値が1かTrueであるカラム名を繋げて一つの文字列にしたもの。

    Examples:

    df['isone'] = df.apply(lambda row: extract_isone(row)[1], axis=1)
    """
    # 宣言
    isone_list = []
    isone_text = ''

    if cols is None:
        # colsがNoneの時はrowのkey全てを対象にする
        cols = list(row.keys())
    else:
        cols = list(cols)
    
    for ii in cols:
        if (row[ii] == 1) or (row[ii] == '1') or (row[ii] == True):
            isone_list += [str(ii)]
            isone_text += str(ii) + sep
    
    # isone_textが2文字以上の時
    if len(isone_text) >= 2:
        # 最後の1文字を消す
        isone_text = isone_text[:-1]
    
    return isone_list, isone_text


def get_cross(df, target=None, cols=[], normalize=False):
    """
    クロス集計をする。pd.crosstabを使用。

    Args:
        df: DataFrame
            クロス集計をしたいDataFrame。
        
        target: str, optional(default=None)
            カウント対象の変数名。目的変数などを想定。
            Noneの時は該当するレコード数をカウントするのみ。
        
        cols: list of str, optional(default=[])
            クロス集計対象となるカラム一覧。dfに存在するカラムのみを想定。
        
        normalize: bool, optional(default=False)
            カウントだけでなく、割合も計算するかどうか。
            Trueの時計算する。targetを1つ指定している時のみ使用可能。
    
    Returns:
        cross_df: DataFrame
            クロス集計を行ったDataFrame。
    """
    tmp = df.copy()
    
    # カテゴリカルな説明変数などをクロス集計のindexに使用するためセット
    cross_idxes = []
    # dfに存在するカラムのみにする
    cols = [str(ii) for ii in cols if str(ii) in tmp.columns.tolist()]
    for col in cols:
        cross_idxes.append(tmp[col])
    
    # 目的変数などをクロス集計のcolumnsに使用するためセット
    cross_cols = []
    if (type(target) == str)and(str(target) in tmp.columns.tolist()):
        cross_cols = [tmp[target]]
    
    # クロス集計を算出
    cross_df = pd.crosstab(index=cross_idxes, columns=cross_cols, margins=True, normalize=False).reset_index(drop=False)
    cross_df = cross_df.rename(columns={'__dummy__': 'all_count'})
    
    if len(cross_cols) == 1:
        cross_df = cross_df.rename(columns=(lambda x: x if not(x in tmp[target].unique().tolist()) else target + '_' + str(x) + '_count'))
    
    # 割合のクロス集計を算出
    if (normalize)and(len(cross_cols) == 1):
        norm_cross_df = pd.crosstab(index=cross_idxes, columns=cross_cols, margins=False, normalize='index').reset_index(drop=False)
        norm_cross_df.drop(labels=cols, inplace=True, axis=1)
        norm_cross_df = norm_cross_df.rename(columns=(lambda x: x if not(x in tmp[target].unique().tolist()) else target + '_' + str(x) + '_ratio'))
        cross_df = pd.merge(cross_df, norm_cross_df, left_index=True, right_index=True, how='outer')
    del cross_df.columns.name
    gc.collect()

    return cross_df


def rfc_gridsearch(data_x, data_y, param_grid=None, evaluation='accuracy', n_cv=5):
    """
    sklearn.ensemble.RandomForestClassifier()の
    GridSearchを行う。

    Args:
        data_x: array-like or sparse matrix of shape = [n_samples, n_features]
            説明変数。
        
        data_y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            目的変数。
        
        param_grid: dict, optional(default=None)
            変化させるパラメータの値。初期値は以下。
            param_grid = [{
                'n_estimators':[i for i in range(10,50,5)],
                'criterion':['gini','entropy'],
                'max_depth':[i for i in range(1,21,1)],
                'min_samples_leaf':[i for i in range(1, 10, 1)],
                'bootstrap':[False],
                'random_state':[57]
            }]
        
        evaluation: str, optional(default='accuracy')
            評価関数。
            評価関数evaluationで指定できるもの: 
                'accuracy'	metrics.accuracy_score
                'balanced_accuracy'	metrics.balanced_accuracy_score	for binary targets
                'average_precision'	metrics.average_precision_score
                'brier_score_loss'	metrics.brier_score_loss
                'f1'	metrics.f1_score	for binary targets
                'f1_micro'	metrics.f1_score	micro-averaged
                'f1_macro'	metrics.f1_score	macro-averaged
                'f1_weighted'	metrics.f1_score	weighted average
                'f1_samples'	metrics.f1_score	by multilabel sample
                'neg_log_loss'	metrics.log_loss	requires predict_proba support
                'precision' etc.	metrics.precision_score	suffixes apply as with 'f1'
                'recall' etc.	metrics.recall_score	suffixes apply as with 'f1'
                'roc_auc'
        
        n_cv: int, optional(default=5)
            grid searchでscoreを出す際に使うcross validationの分割数。
    
    Returns:
        gs: instance of GridSearchCV
            grid searchのインスタンス。
    
    Examples:
    
    evaluations_list = ['accuracy', 'f1', 'precision', 'recall']
    for eval_func in evaluations_list:
        gs = rfc_gridsearch(data_x, data_y, evaluation=eval_func)
        print('__________________________________________________________________  ')
    
    Remarks:
        RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        GridSearchCV:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    # ランダムフォレストのインスタンスを作成
    rfc_gs = RandomForestClassifier(random_state=57)

    # ハイパーパラメータ値のリスト
    if type(param_grid) == dict:
        param_grid = [param_grid]
    else:
        param_grid = [{
            'n_estimators':[i for i in range(10,50,5)],
            'criterion':['gini','entropy'],
            'max_depth':[i for i in range(1,21,1)],
            'min_samples_leaf':[i for i in range(1, 10, 1)],
            'bootstrap':[False],
            'random_state':[57]
        }]

    # ハイパーパラメータ値のリストparam_gridを指定し、
    # グリッドサーチを行うGridSearchCVクラスをインスタンス化
    gs = GridSearchCV(
        estimator = rfc_gs,
        param_grid = param_grid,
        scoring = evaluation,
        cv = n_cv,
        n_jobs = 5
    )

    # fitさせる
    gs = gs.fit(data_x, data_y)

    # 評価関数
    print('evaluation:', evaluation, '  ')
    
    # モデルの最良スコアを出力
    print('best score:', gs.best_score_, '  ')

    # 最良スコアとなるパラメータ値を出力
    print('best params:', gs.best_params_, '  ')

    return gs


def extract_cols(input_df, text=''):
    """
    特定のテキストを含むカラムのみを抽出して情報を表示する。
    似たような名前のカラムが多くある時に有効。

    Args:
        input_df: DataFrame
            分析対象のDataFrame。
        
        text: str, optional(default='')
            検索したいテキスト。
            このテキストを含むカラムについての情報を表示する。
    """
    num = len(input_df)
    
    # カラム一覧
    cols = input_df.columns.tolist()
    
    # textを含むカラムのリスト
    ex_cols = [ii for ii in cols if text in ii]
    
    output_df = input_df[ex_cols]
    
    print('欠損率(%)-------------------------------------------------------------')
    print(pro_round(output_df.isna().sum() / num * 100, 3))
    print('--------------------------------------------------------------------------')

    # dfを表示する
    # display(output_df)
    
    return output_df


class ResultManager():
    """
    Attributes:
        target: str
            目的変数名。

        result_df: DataFrame
            結果をまとめたDataFrame。

        unique_df: DataFrame
            分類モデルでのみ使用可能。
            各変数の値の組み合わせごとに目的変数が1である確率をまとめたDataFrame。

        cm_df: DataFrame
            分類モデルでのみ使用可能。
            混同行列。ConfusionMatrix。

        model_type: str or None
            モデルのタイプ。分類か回帰か。
            'clf'か'reg'を指定。
    """
    target = 'target'
    result_df = None
    unique_df = None
    score_df = None
    cm_df = None
    model_type = None


    def __init__(self, model_type=None):
        """
        コンストラクタ。

        Args:
            model_type: str, optional(default=None)
                モデルのタイプ。分類か回帰か。
                'clf'か'reg'を指定。
        """
        self.model_type = None
        if (model_type is None) or (model_type == 'clf') or (model_type == 'reg'):
            self.model_type = model_type
        else:
            print('model type error')
            return
        # print('model type is ', self.model_type)
    

    def set_clf_result(self, data_x, data_y, pred_y, prob_y, target='target'):
        """
        分類モデルの結果をまとめるためのデータをセットする。

        Args:
            data_x: DataFrame
                説明変数。

            data_y: 1d array-like, or label indicator array / sparse matrix
                目的変数。予測値に対する実測値のこと。

            pred_y: 1d array-like, or label indicator array / sparse matrix
                予測値。

            prob_y: 1d array-like, or label indicator array / sparse matrix
                目的変数の値が1を取る確率。

            target: str, optional(default='target')
                目的変数名。
        """
        self.target = target
        result_df = data_x.copy()
        self.features = data_x.columns.tolist()
        result_df[target] = data_y
        result_df['pred_y'] = pred_y
        result_df['pred_prob'] = prob_y.T[1]

        print('______________________________')
        print('evaluation score')
        print()
        score_df, cm_df = print_clf_score(data_y, pred_y)
        print('______________________________')

        unique_df = self.result_to_unique(result_df, cols_list=self.features)
        
        self.result_df = result_df
        self.score_df = score_df
        self.cm_df = cm_df
        self.unique_df = unique_df

        self.model_type = 'clf'
        gc.collect()
    

    def set_reg_result(self, data_x, data_y, pred_y, target='target'):
        """
        回帰モデルの結果をまとめるためのデータをセットする。

        Args:
            data_x: DataFrame
                説明変数。

            data_y: 1d array-like, or label indicator array / sparse matrix
                目的変数。予測値に対する実測値のこと。

            pred_y: 1d array-like, or label indicator array / sparse matrix
                予測値。

            target: str, optional(default='target')
                目的変数名。
        """
        self.target = target
        result_df = data_x.copy()
        self.features = data_x.columns.tolist()
        result_df[target] = data_y
        result_df['pred_y'] = pred_y

        print('______________________________')
        print('evaluation score')
        print()
        score_df = print_reg_score(data_y, pred_y)
        print('______________________________')
        
        self.result_df = result_df
        self.score_df = score_df

        self.model_type = 'reg'
        gc.collect()


    def result_to_unique(self, result_df, cols_list):
        """
        result_dfからunique_dfを作成する。

        Args:
            result_df:
                uniqueを考える対象のDataFrame。

            cols_list: list of str
                uniqueを考える対象のカラム。
        """
        cols_df_list = []
        for col in cols_list:
            cols_df_list.append(result_df[col])
        
        cross_df = pd.crosstab(margins=True, index=cols_df_list, columns=all).reset_index()
        cross_df = cross_df[cols_list + ['All']].rename(columns={'All': 'count'})
        
        del cross_df.columns.name # indexを直す
        cross_df[cols_list] = cross_df[cols_list].astype(str) # 型をstrにする
        
        # uniqueのlistを作る
        unique_df = result_df.drop_duplicates(subset=cols_list)
        
        unique_df[cols_list] = unique_df[cols_list].astype(str) # 型をstrにする
        
        # マージする
        unique_df = pd.merge(unique_df, cross_df, on=cols_list, how='left')
        
        # ソート
        unique_df = unique_df.sort_values(by=['pred_prob', 'count'], ascending=[False, False]).reset_index(drop=True)
        unique_df['count_cumsum'] = np.cumsum(unique_df['count'].values)
        unique_df['count_cumratio'] = unique_df['count_cumsum'] / unique_df['count'].values.sum()
        unique_df['summary'] = unique_df.apply(lambda row: extract_isone(row, cols_list), axis=1)
        
        return unique_df
    

    def get_result(self):
        """
        modelの評価値など様々な結果情報を返す。

        Returns:
            ret: tuple of DataFrame or bool
                model_typeがclfの時
                    result_df, unique_df, score_df, cm_df
                model_typeがregの時
                    result_df, score_df
                それ以外の時
                    False
        """
        if self.model_type == 'clf':
            ret = self.result_df, self.unique_df, self.score_df, self.cm_df
        elif self.model_type == 'reg':
            ret = self.result_df, self.score_df
        else:
            print('model type error')
            ret = False
        return ret


class RandomForestManager():
    """
    ランダムフォレストの諸々を管理する。

    Attributes:
        rfc: instance of RandomForestClassifier
            ランダムフォレストモデルのインスタンス。

        target: str
            目的変数。

        features: list of str
            特徴量。
        
        estimators: dict
            木々。
        
        is_trained: bool
            学習済みフラグ
    
    Examples:
    
    rfc_args = {
        'bootstrap': True,
        'criterion': 'gini',
        'max_depth': 11,
        'min_samples_leaf': 5,
        'n_estimators': 100,
        'random_state': 57
    }
    rfm = RandomForestManager(args=rfc_args, target='survived')
    """
    rfc = None
    rfc_attributes = None

    # 目的変数
    target = 'target'

    # 特徴量
    features = []
    """
    # 保持しないことにする
    # 教師データ
    train_df = None
    train_x = None
    train_y = None
    """
    train_shape = (0, 0)

    # 寄与率
    importances = None

    # 木々
    estimators = None

    # 学習済みフラグ
    is_trained = False


    def __init__(self, args=None, target='target', features=[]):
        """
        コンストラクタ。

        Args:
            args: dict
                ランダムフォレストの引数。

            target: str
                目的変数。

            features: list of str
                特徴量。
        """
        self.rfc = None
        if args is None:
            self.rfc = RandomForestClassifier()
            self.rfc_attributes = dict(self.rfc.__dict__)
            print('rfc model is initialized')
        elif type(args) == dict:
            self.rfc = RandomForestClassifier(**args)
            self.rfc_attributes = dict(self.rfc.__dict__)
        else:
            print('warning: args is not dictionary')
            print('rfc model is initialized')
            self.rfc = RandomForestClassifier()
            self.rfc_attributes = dict(self.rfc.__dict__)
            print('rfc model is initialized')
            
        self.target = target
        self.features = features
        self.importances = None
        self.is_trained = False


    def set_rfc_args(self, args):
        """
        ランダムフォレストの属性を設定する。

        Args:
            args: dict
                ランダムフォレストの引数。
        """
        if type(args) == dict:
            self.rfc = RandomForestClassifier(**args)
            self.rfc_attributes = dict(self.rfc.__dict__)
            print('rfc model is initialized')
            print('rfc args are set')
            self.is_trained = False
        else:
            print('error: type of args is not dictionary')


    def train(self, train_df):
        """
        学習する。

        Args:
            train_df: DataFrame
                教師データ。
        """
        train_cols = train_df.columns.tolist()
        if (type(self.target)==str)and(len(self.features)>=1):
            if (self.target in train_cols)and(set(self.features) <= set(train_cols)):
                # self.train_df = train_df
                # self.train_x = self.train_df[self.features]
                # self.train_y = self.train_df[self.target]
                # self.rfc.fit(self.train_x, self.train_y)
                train_x = train_df[self.features]
                train_y = train_df[self.target]
                self.rfc.fit(train_x, train_y)
                print('model is trained')
                # 寄与率
                self.importances = self.rfc.feature_importances_
                # 木々
                self.estimators = self.rfc.estimators_
                # 学習済みフラグ
                self.is_trained = True
            else:
                print('train error: there is an error in columns')
        else:
            print('train error: there is an error in columns')

    
    def train_xy(self, train_x, train_y):
        """
        学習する。

        Args:
            train_x: DataFrame
                教師データの特徴量部分。

            train_y: 1d array-like, or label indicator array / sparse matrix
                教師データの目的変数部分。
        """
        if len(train_x) == len(train_y):
            # self.train_x = train_x
            # self.train_y = train_y
            # self.train_df = pd.concat([self.train_x, self.train_y], axis=1, join='outer')
            self.features = train_x.columns.tolist()
            self.rfc.fit(train_x, train_y)
            self.is_trained = True
            print('model is trained')
            # 寄与率
            self.importances = self.rfc.feature_importances_
            # 木々
            self.estimators = self.rfc.estimators_
            # 学習済みフラグ
            self.is_trained = True
        else:
            print('train error: length of x and length of y is different')
    

    def predict(self, data_x):
        """
        data_xからyを予測する。
        pred_y, prob_y

        Args:
            data_x: DataFrame
                説明変数。
        
        Returns:
            ret: tuple of DataFrame
                pred_y, prob_y
                pred_y:
                    目的変数の予測値。
                prob_y:
                    目的変数が1である確率。
        """
        pred_y = self.rfc.predict(data_x)
        prob_y = self.rfc.predict_proba(data_x)
        ret = pred_y, prob_y
        gc.collect()
        return ret
    

    def validate(self, data_x, data_y):
        """
        検証。
        result_df, unique_df, score_df, cm_df

        Args:
            data_x: DataFrame
                説明変数。

            data_y: 1d array-like, or label indicator array / sparse matrix
                目的変数。
        
        Returns:
            ret: tuple of DataFrame
                result_df, unique_df, score_df, cm_df
        """
        pred_y = self.rfc.predict(data_x)
        prob_y = self.rfc.predict_proba(data_x)

        rm = ResultManager(model_type='clf')
        rm.set_clf_result(data_x, data_y, pred_y, prob_y, target=self.target)
        result_df, unique_df, score_df, cm_df = rm.get_result()
        ret = result_df, unique_df, score_df, cm_df
        gc.collect()
        return ret


    def get_importance(self, figsize=(12, 9)):
        """
        寄与率を表すDataFrameを返す。

        Args:
            figsize: tuple of int, optional(default=(12, 9))
                グラフの大きさ。横と縦の大きさ。
        
        Returns:
            importance_df: DataFrame
                寄与率を表すDataFrame。
                columns: feature, importance
        """
        # importances
        importance_df = pd.DataFrame({'feature': self.features, 'importance': self.importances}).sort_values(by='importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        importance_df.sort_values(by='importance', ascending=True).plot(x='feature', y='importance', kind='barh', figsize=figsize)
        return importance_df
    

    def visualize_tree(self, num_tree=0, class_names=['0', '1'], tree_file='rfc_tree', dir_path='./'):
        """
        木の1つを可視化する。

        Args:
            num_tree: int, optional(default=0)
                何番目の木を可視化するか。0始まり。
            
            class_names: list of str, optional(default=['0', '1'])
                木のクラスの名前。
            
            tree_file: str, optional(default='rfc_tree')
                木のファイル名。
            
            dir_path: str, optional(default='./')
                木のファイルを置くディレクトリ。
        """
        estimators = self.estimators
        n_estimators = len(estimators)

        # 番号の指定が正しいか判定する
        if (0 <= num_tree) and (num_tree < n_estimators):
            file_name = dir_path + tree_file + str(num_tree) + '.png'
            dot_data = tree.export_graphviz(estimators[num_tree], # 決定木オブジェクトを一つ指定する
                                            out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                            filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                            node_ids=True, # ノード番号出力?
                                            # proportion=True, # パーセンテージ表記
                                            rounded=True, # Trueにすると、ノードの角を丸く描画する。
                                            feature_names=self.features, # これを指定しないとチャート上で特徴量の名前が表示されない
                                            class_names=class_names, # これを指定しないとチャート上で分類名が表示されない
                                            special_characters=True # 特殊文字を扱えるようにする
                                            )
            graph = pdp.graph_from_dot_data(dot_data)

            # 木を画像ファイルとして出力
            graph.write_png(file_name)

            # 木を表示する
            Image(file_name)
        else:
            print('error: num_tree < n_estimators')
            print('n_estimators is ', len(estimators))


