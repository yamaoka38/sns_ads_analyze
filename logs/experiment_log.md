# 概要
コードの分析結果やメモを記録

# 履歴

## 分析内容 Ver2.0
### 実施内容
- 4つのCSVの統合
- event_typeをワンホットエンコーディング
- 各特徴量の分布を確認

### 今後の予定
- データの前処理（特徴量）
- KMeans法を用いてクラスタリングを実施
- LightGBM、ランダムフォレストなどでClick・Purchseそれぞれの行動予測を実施

クラスタリング、行動予測はユーザー属性を示す特徴量、広告属性を示す特徴量、ユーザー・広告属性を統合した特徴量による3パターンを実施。
また、ユーザーセグメントｘ広告セグメントによる行動予測も実施予定

## 分析結果 Ver2.0
特徴量の分布は下記の通り
- 性別はやや男性寄り
- 年齢は20代がピーク。40代以上は少数
- 曜日による傾向さは見られない
- 国はアメリカ・イギリスが上位
- 広告タイプはストーリーズがやや多めだが、カルーセル・イメージ・動画それぞれ一定数あり
- 広告プラットフォームはFacebookとInstagramの2種類。Facebookが多め

## 分析内容 Ver5.0
### 実施内容
- 簡易特徴量にてclick予測のスクリプトを実行
    モデル：ロジスティック回帰・ランダムフォレスト・LightGBM
    評価指標：AUC・Logloss
    特徴量："ad_platform","ad_type","user_gender","user_age",
    "click","hour_sin","hour_cos","user_cluster_id","ad_cluster_id"

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

## 分析内容 Ver5.1
### 実施内容（Ver5.0からの変更点）
- 特徴量を追加してclick予測のスクリプトを実行
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

## 分析内容 Ver5.2
### 実施内容（Ver5.1からの変更点）
・クラスターIDを除外したモデリング
・クラスターIDのみを使ったモデリング
を実施

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