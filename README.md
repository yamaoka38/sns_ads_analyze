# exp/ad/clustering (v1.4)

## 🧭 目的
広告特徴量でのクラスタリングを実施。

## 🔄 実施内容
- 特徴量に以下を使用
カテゴリカル変数（ワンホットエンコーディング）："ad_platform","ad_type","target_gender","target_age_group","target_interests"
標準化："duration_days","total_budget"
- ad_id毎の平均CTR、CVRを特徴量として追加
- カテゴリ数の多い"target_interests"を2次元でPCAを行ってからクラスタリングを実施
- DBI、CHを評価指標として使用
- k=2,4,6,7 で最適なクラスタ数を判断


## ⚙️ 使用データ
- ads.csv
- campaigns.csv
- events.csv

## 📊 現状の結果
- kの増加に伴い、DBI,CHともに低下
- 詳細スコアは以下の通り
k	DBI	CH
2	2.250127113	54350.69798
4	2.170420295	40597.2696
6	2.106792925	34300.27087
7	2.007489019	31549.62431

- 散布図の結果からはk=6 が一番良さそう

## 🚧 今後の課題
### 処理内容
- k=5，6 でそれぞれクラスタリングした散布図と特徴量の可視化を確認し、最適なクラスタ数を最終判断

## 📝 備考
- branch: `feature/clustering-user-v3.1-draft`
- status: **検証中**
- 最終マージ予定：10/17 ver3