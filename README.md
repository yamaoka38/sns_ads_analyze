# exp/ad/clustering (v1.0)

## 🧭 目的
広告特徴量でのクラスタリングを実施。

## 🔄 実施内容
- 特徴量に以下を使用
カテゴリカル変数（ワンホットエンコーディング）："ad_platform","ad_type","target_gender","target_age_group","target_interests"
標準化："duration_days","total_budget"
- DBI、CHを評価指標として使用
- k=6 で各クラスタの特徴量の傾向と、平均CTR・CVR・CTVRを算出

## ⚙️ 使用データ
- ads.csv
- campaigns.csv
- events.csv

## 📊 現状の結果
- DBI：2.581、CH：32467.875 でクラスタの分離がうまく出来ていない
- クラスタごとのCTR差は見られず、CVR差もわずか
- 特徴量として、"ad_platform"は明確にクラスタが分かれている。
- "ad_type"でカルーセルの比率が大きいクラスタはCVRが低めの傾向
- その他、傾向差の無い特徴量はなかった。（ユーザー特徴量ではインタレストが傾向差がなかったが、広告側では傾向差があり影響はありそう。）

## 🚧 今後の課題
### 処理内容
- ワンホットエンコーディングでのカテゴリが多いため、まずは"target_interests"を頻度エンコーディングに変えて様子見
- 特徴量が固まったらkの値を2~10の範囲で検証

## 📝 備考
- branch: `feature/clustering-user-v3.1-draft`
- status: **検証中**
- 最終マージ予定：10/17 ver3