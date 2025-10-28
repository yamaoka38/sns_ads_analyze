# sns_ads_analyze.py(Ver5.2)

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
- 簡易特徴量にてclick予測のスクリプトを実行
    モデル：ロジスティック回帰・ランダムフォレスト・LightGBM
    評価指標：AUC・Logloss

- 目的変数："click"
- 特徴量：
・パターン1：クラスタIDを除いて実施（model_click_outid.py）
"day_of_week", ⇒周期エンコーディング
"ad_platform",
"ad_type",
"target_gender",
"target_interests",
"duration_days",
"total_budget",
"user_gender",
"user_age",
"month",
"day",
"day_from_start",
"hour_sin",
"hour_cos",
"art",
"fashion",
"finance",
"fitness",
"food",
"gaming",
"health",
"lifestyle",
"news",
"photography",
"sports",
"technology",
"travel",
"avg_ctr"

・パターン2：クラスタIDのみ（model_click_idonly.py）
"user_cluster_id",
"ad_cluster_id"

・パターン3：パターン1と2の合算（model_click.py）
※Ver5.0からmonthとdayの標準化を追加


## 分析結果
- 現時点では、AUC・Loglossの評価指標は以下の通り
パターン1（クラスタID除外）
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
0      LightGBM          0.8858         0.5062              0.2963
1  RandomForest          1.0000         0.5028              0.0816
2        LogReg          0.5115         0.4967              0.3365

test_Logloss_mean  fit_time_mean  score_time_mean
0             0.3390        17.6834           0.3240
1             0.3461       139.7476          12.6265
2             0.3367         0.9416           0.0865

パターン2（クラスタIDのみ）
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
0      LightGBM          0.5157         0.5077              0.3365
1  RandomForest          0.5157         0.5077              0.3365
2        LogReg          0.5026         0.4966              0.3366

test_Logloss_mean  fit_time_mean  score_time_mean
0             0.3366        31.8236           0.6238
1             0.3366        22.5926           1.9698
2             0.3366         0.1035           0.0287

パターン3（パターン1＋2）
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
0      LightGBM          0.8902         0.5043              0.2957
1  RandomForest          1.0000         0.5010              0.0816
2        LogReg          0.5120         0.4968              0.3365

test_Logloss_mean  fit_time_mean  score_time_mean
0             0.3391        22.5395           0.5736
1             0.3466       141.6166          14.0647
2             0.3367         2.5252           0.1530



クラスタIDのみではAUCが低く、行動予測には寄与していない状態。
逆にクラスタIDを除外してもほとんどスコアは変わらず。
現時点のクラスタリングでは行動予測には直接寄与はしていないことが分かった。
そもそもがサンプルデータのため、明確に効く特徴量が存在しない可能性もゼロではないが、特徴量の作り方を見直していく。

### 今後の予定
- 特徴量を見直すため、改めて既存特徴量の分布やCTRなどを確認

## バージョン履歴
**Ver5.2(2025-10-28)**:click予測のスクリプトの特徴量を変更
**Ver5.1(2025-10-27)**:click予測のスクリプトの特徴量を追加
**Ver5.0(2025-10-27)**:click予測のスクリプトを追加
**Ver4(2025-10-24)**:ファイル・ディレクトリ構成を変更（各スクリプトをモジュール化）
**Ver3(2025-10-18)**:全レコードを対象にデータの前処理を実行。その後テストデータと訓練データに分割
**Ver2(2025-10-17)**:コーディングの仕様変更（#%%を用いた記述に変更。特徴量の分布確認などを実行）
**Ver1(2025-10-15)**:初回（データの読み込み・統合・データの前処理）

## ライセンス
BSD 3-Clause License