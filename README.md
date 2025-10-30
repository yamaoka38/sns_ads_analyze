# sns_ads_analyze.py(Ver5.4)

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
- クリック予測モデルにおいて過学習抑制のためのパラメータ調整を実施
    - 検証1：LightGBM＆ランダムフォレストをmax_depth 制限なし⇒8 にした場合
    - 検証2：LightGBMをmax_depth 制限なし⇒8かつnum_leaves=63⇒30に変更、ランダムフォレストをmax_depth 8 ⇒ 4 に変更
    - 検証3：LightGBMをmax_depth 制限なし⇒8かつnum_leaves=30⇒15に変更、ランダムフォレストをmax_depth 8 ⇒ 4 に変更


## 分析結果
- 検証1：LightGBM＆ランダムフォレストをmax_depth 制限なし⇒8 にした場合	
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
1      LightGBM          0.8647         0.5036              0.2989
0  RandomForest          0.6818         0.5120              0.3323
2        LogReg          0.5120         0.4968              0.3365

test_Logloss_mean  fit_time_mean  score_time_mean
1             0.3397        21.6187           0.5366
0             0.3366        61.6760           2.9194
2             0.3367         1.2292           0.0825


- 検証2：LightGBMをmax_depth 制限なし⇒8かつnum_leaves=63⇒30に変更、ランダムフォレストをmax_depth 8 ⇒ 4 に変更
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
1      LightGBM          0.7793         0.5062              0.3162
0  RandomForest          0.5423         0.5148              0.3362
2        LogReg          0.5120         0.4968              0.3365

test_Logloss_mean  fit_time_mean  score_time_mean
1             0.3381        14.4003           0.6605
0             0.3365        35.3673           1.5161
2             0.3367         1.2105           0.0799

- 検証3：LightGBMをmax_depth 制限なし⇒8かつnum_leaves=30⇒15に変更、ランダムフォレストをmax_depth 8 ⇒ 4 に変更
model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
1      LightGBM          0.6950         0.5095              0.3258
0  RandomForest          0.5423         0.5148              0.3362
2        LogReg          0.5120         0.4968              0.3365

test_Logloss_mean  fit_time_mean  score_time_mean
1             0.3373        13.2820           0.6330
0             0.3365        38.9410           2.5069
2             0.3367         1.1205           0.0701

⇒検証1~3のいずれにおいてもパラメータを絞ることで、LightGBM、ランダムフォレストでのtrain_AUCは減少し、過学習は抑制の方向に向かったが、それによるtest_AUCの改善はほとんど見られなかった。
パラメータ調整では本質的な改善に至らず、やはり特徴量の改善が必要と思われる。

### 今後の予定
#### クラスタリング
- CTRにおける広告クラスタの影響に寄与する要因を可視化するため、興味関心を除いたVer・性別を除いたVer・広告プラットフォームを除いたVerでのCTRの変化をそれぞれ検証

#### 行動予測モデル構築
- その上で特徴量の加工を検討

## バージョン履歴
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