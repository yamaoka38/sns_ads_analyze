# sns_ads_analyze.py(Ver4)

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

## 分析結果
- 現時点では、AUC・Loglossの評価指標は以下の通り

model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
0      LightGBM          0.7174         0.5015              0.3208
1  RandomForest          0.9507         0.4944              0.1705
2  ロジスティック回帰          0.5058         0.4934              0.3366

test_Logloss_mean  fit_time_mean  score_time_mean
0             0.3394        22.4383           0.9994
1             0.9273        93.1737          29.5390
2             0.3367         0.2916           0.0509

ツリー系モデルではtrainデータに対しての学習は良好だが、testデータに対しては不十分。
ロジスティック回帰ではtrain、testデータの差はなく過学習はないが、そもそも学習が進んでない。
LoglossではLightGBM、ロジスティック回帰ともに悪くないスコアだが、そもそもclickが少ない影響といえそう。
まずは予測に影響を与える特徴量を発見するため、特徴量の見直しが必要。

### 今後の予定
- 特徴量を見直して予測モデルを再構築

## バージョン履歴
**Ver5(2025-10-27)**:click予測のスクリプトを追加
**Ver4(2025-10-24)**:ファイル・ディレクトリ構成を変更（各スクリプトをモジュール化）
**Ver3(2025-10-18)**:全レコードを対象にデータの前処理を実行。その後テストデータと訓練データに分割
**Ver2(2025-10-17)**:コーディングの仕様変更（#%%を用いた記述に変更。特徴量の分布確認などを実行）
**Ver1(2025-10-15)**:初回（データの読み込み・統合・データの前処理）

## ライセンス
BSD 3-Clause License