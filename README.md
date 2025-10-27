# sns_ads_analyze.py(Ver5.1)

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
"user_cluster_id",
"ad_cluster_id",
"avg_ctr"


## 分析結果
- 現時点では、AUC・Loglossの評価指標は以下の通り
          model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
0      LightGBM          0.8902         0.5043              0.2957
1  RandomForest          1.0000         0.5011              0.0816
2  ロジスティック回帰      0.5120         0.4971              0.3365

   test_Logloss_mean  fit_time_mean  score_time_mean
0             0.3391        26.6097           0.6314
1             0.3465       173.9100          22.1270
2             0.3367         2.7449           0.0771


特徴量の追加で全体的にtrain_AUCのスコアが向上したが、test_AUCの伸びは微増程度
⇒過学習が進んだ印象。
LightGBMのtest_logloss は 0.9273⇒0.3465に大きく改善。


### 今後の予定
- 特徴量を再度見直して予測モデルを再構築

## バージョン履歴
**Ver5.1(2025-10-27)**:click予測のスクリプトの特徴量を追加
**Ver5(2025-10-27)**:click予測のスクリプトを追加
**Ver4(2025-10-24)**:ファイル・ディレクトリ構成を変更（各スクリプトをモジュール化）
**Ver3(2025-10-18)**:全レコードを対象にデータの前処理を実行。その後テストデータと訓練データに分割
**Ver2(2025-10-17)**:コーディングの仕様変更（#%%を用いた記述に変更。特徴量の分布確認などを実行）
**Ver1(2025-10-15)**:初回（データの読み込み・統合・データの前処理）

## ライセンス
BSD 3-Clause License