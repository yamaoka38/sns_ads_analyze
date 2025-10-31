# sns_ads_analyze.py(Ver5.6)

## 概要
SNS広告のクリック・購入データを用いてユーザー・広告のセグメント化を行い、クリック・購入の行動予測を実施。
セグメント別の傾向把握や予測モデルの構築から、成果改善の示唆出しを目的とする。
尚、Kaggleの公開データセットを用いている。
データは若年層向けの総合ファッションECのものと仮定して進める。

## ディレクトリ構成
project/
├─ rawdata/ # 元データ（ads.csv, users.csvなど）
├─ hub/ # ハブスクリプト（sns_ads_pipline.py）、yaml
├─ modules/ # 個別モジュールのPythonスクリプト
├─ outputs/ # 出力ファイル（CSV、画像）
├─ logs/ # 各検証の結果を記録
└─ README.md

## 使用技術
- Python 3.13.7
- numpy / pandas / scikit-learn / matplotlib / seaborn
- Cursor
- Git / GitHub

## 実行方法
sns_ads_pipline.py を実行

## 分析内容
### 実施内容
- 情報がリークしている恐れがある広告ID毎の平均CTRを特徴量から除いて検証
    - 検証1：LightGBMをmax_depth=8かつnum_leaves=15、ランダムフォレストをmax_depth=4
    - 検証2：LightGBMをmax_depth=-1かつnum_leaves=63、ランダムフォレストをmax_depth=None

## 分析結果
- 各検証の結果は以下の通り
    - 検証1：LightGBMをmax_depth=8かつnum_leaves=15、ランダムフォレストをmax_depth=4
            model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
        0  RandomForest          0.5511         0.5048              0.3363
        1      LightGBM          0.7011         0.5036              0.3263
        2        LogReg          0.5121         0.4970              0.3365

        test_Logloss_mean  fit_time_mean  score_time_mean
        0             0.3366        37.0677           1.6104
        1             0.3375        14.8037           0.5990
        2             0.3367         1.1974           0.0849

    - 検証2：LightGBMをmax_depth=-1かつnum_leaves=63、ランダムフォレストをmax_depth=None
            model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
        0  RandomForest          1.0000         0.5023              0.0816
        1      LightGBM          0.8920         0.5021              0.2966
        2        LogReg          0.5121         0.4970              0.3365

        test_Logloss_mean  fit_time_mean  score_time_mean
        0             0.3455       159.0806          22.4799
        1             0.3391        25.6162           0.7162
        2             0.3367         1.1338           0.0771

- 平均CTRを除いても目立った変化は無かった
- 改めて上記検証1、2それぞれの条件でSHAP値を出したが、平均CTRの影響を受けている広告クラスタIDの寄与度が上昇
    - 以後は過学習を抑制している検証1のパラメータ設定をベースに進める

### 今後の予定
#### 行動予測モデル構築
- 引き続き特徴量を検証
    - 検証1：寄与度の高い年齢をカテゴリ分けして検証
    - 検証2：カテゴリ数の多い興味関心(ターゲット・ユーザー)をPCAで次元削減して検証

## バージョン履歴
**Ver5.6(2025-10-31)**:特徴量から平均CTRを除いてモデルの検証
**Ver5.5(2025-10-31)**:SHAP値で各特徴量のモデルへの寄与度を可視化
**Ver5.4(2025-10-31)**:ツリー系モデルのパラメータを変更し、過学習抑制の検証
**Ver5.3(2025-10-30)**:ユーザークラスタIDｘ広告クラスタIDのCTR・CVRを可視化
**Ver5.2(2025-10-28)**:click予測のスクリプトの特徴量を変更
**Ver5.1(2025-10-27)**:click予測のスクリプトの特徴量を追加
**Ver5.0(2025-10-27)**:click予測のスクリプトを追加
**Ver4(2025-10-24)**:ファイル・ディレクトリ構成を変更（各スクリプトをモジュール化）
**Ver3(2025-10-18)**:全レコードを対象にデータの前処理を実行。その後テストデータと訓練データに分割
**Ver2(2025-10-17)**:コーディングの仕様変更（#%%を用いた記述に変更。特徴量の分布確認などを実行）
**Ver1(2025-10-15)**:初回（データの読み込み・統合・データの前処理）

## ライセンス
BSD 3-Clause License