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

## 分析内容 Ver5.3
### 実施内容（Ver5.2からの変更点）
- ユーザークラスタIDと広告クラスタIDの掛け合わせによるCTR・CVRを可視化
    - 実行スクリプト：cluster_id_matrix.py

- CTR
CTR	ad_id				
user_cluster_id	0	1	2	3	4
0	10.39%	9.89%	11.40%	10.54%	10.03%
1	10.39%	10.22%	11.85%	10.69%	9.90%
2	10.59%	10.21%	11.35%	10.90%	9.66%
3	10.17%	10.57%	11.60%	10.77%	9.80%
4	10.59%	9.92%	11.78%	10.32%	10.63%
5	10.39%	10.38%	11.40%	10.37%	10.02%

CVR
CVR	ad_id				
user_cluster_id	0	1	2	3	4
0	6.87%	3.42%	4.61%	2.91%	7.18%
1	6.76%	3.36%	4.85%	3.78%	7.05%
2	5.46%	3.42%	4.86%	3.67%	5.63%
3	5.28%	4.35%	4.76%	3.68%	7.05%
4	5.74%	3.89%	5.31%	3.16%	8.09%
5	6.45%	5.21%	5.23%	3.99%	4.37%


CTR・CVRともにユーザークラスタIDよりも広告クラスタIDの影響が大きいことが分かった。
CTRでは広告クラスタID 2、CVRでは広告クラスタID0、5が高い傾向がみられる。

- 広告クラスタID 2の特徴
    - Instagramの比重が高い
    - 男性比率が高い
    - 興味関心でFitness・Sportsの比率が高い

- 広告クラスタID 0、5の共通の特徴
    - 広告タイプでストーリーズの比率が高い（プラットフォームはFacebookの比重が高い）

### 今後の予定
#### クラスタリング
- CTRにおける広告クラスタの影響に寄与する要因を可視化するため、興味関心を除いたVer・性別を除いたVer・広告プラットフォームを除いたVerでのCTRの変化をそれぞれ検証

#### 行動予測モデル構築
- ツリー系モデルの過学習をまずは抑えるため、パラメータを調整してAUCの変化を検証
- その上で特徴量の加工を検討

## 分析内容 Ver5.4
### 実施内容（Ver5.3からの変更点）
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
- 特徴量の加工を検討

## 分析内容 Ver5.5
### 実施内容（Ver5.4からの変更点）
- クリック予測モデル（LightGBM）において、SHAP値を出力して予測への各特徴量の寄与度を可視化 (model_click_LGBM_SHAP.py)

## 分析結果
- 寄与度としては広告IDの平均CTR、ユーザー年齢、時刻、日付、開始日からの経過日数の影響が高くみられた。
    - 平均CTRを入れてしまったことで情報がリークしてしまい過学習している可能性が考えられる。
- ユーザー年齢は高年齢層がCTRを高める方向にも下げる方向にも影響しており、単純な年齢だけでCTRが高い・低いとは言い切れない
    - 興味関心カテゴリにも同じような傾向がみられており、年齢や興味関心の掛け合わせ寄っても変化があるかもしれない。 

### 今後の予定
#### 行動予測モデル構築
- まずは、情報リークしている平均CTRを除外して再度モデル構築を進める

## 分析内容 Ver5.6
### 実施内容（Ver5.5からの変更点）
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

## 分析内容 Ver5.7
### 実施内容（Ver5.6からの変更点）
- user_ageを4分位法でカテゴリして検証

## 分析結果
- 検証の結果は以下の通り
        model  train_AUC_mean  test_AUC_mean  train_Logloss_mean  \
        0      LightGBM          0.7016         0.5052              0.3265
        1  RandomForest          0.5510         0.5040              0.3363
        2        LogReg          0.5121         0.4966              0.3365

        test_Logloss_mean  fit_time_mean  score_time_mean
        0             0.3374        12.8244           0.6133
        1             0.3366        37.7440           1.7862
        2             0.3367         1.3787           0.0908

- 年齢をカテゴリ化してもスコア改善は見られなかった
    ⇒年齢の処理を元に戻して別軸検証
- SHAP値においては各年齢カテゴリの影響は見られなかった。
    - 日付、開始日からの経過日数、時刻、配信日数など日時データの寄与度が高い傾向がみられた

### 今後の予定
#### 行動予測モデル構築
- 引き続き特徴量を検証
    - 検証2：カテゴリ数の多い興味関心(ターゲット・ユーザー)をPCAで次元削減して検証
    - 検証3：広告の掲載日数（日付―キャンペーン開始日）を加え検証

## 分析内容 Ver-
### 実施内容（Ver5.7からの変更点）
- merge_and_pretreatment.pyに全体実績の算出コードを追加

## 分析結果
- 結果は以下の通り
    imps:403967
    clicks:42545
    cvs:2050
    costs:2535923.7800000003
    ctr:0.10531800864922135
    cvr:0.04818427547302856
    cpm:6277.551829728667
    cpc:59.60568292396287
    cpa:1237.0359902439025

## 分析内容 Ver5.8
### 実施内容（Ver5.7からの変更点）
- クラスタリングにサンプル1万レコードでのシルエットスコアを追加

## 分析結果
- 結果は以下の通り
    ユーザー軸：0.139
    広告軸：0.128

### 今後の予定
#### 全体
- テストデータと学習データの分割をクリック比率ではなく、時系列順で分ける

#### クラスタリング
- ユーザーID単位、広告ID単位でのクラスタリングを実施
- シルエットスコアも含めたうえでKの最適数を評価

#### 行動予測モデル構築
- 引き続き特徴量を検証
    - 検証2：カテゴリ数の多い興味関心(ターゲット・ユーザー)をPCAで次元削減して検証
    - 検証3：広告の掲載日数（日付―キャンペーン開始日）を加え検証
    - 検証4：再構築したクラスタリングIDを追加して検証

