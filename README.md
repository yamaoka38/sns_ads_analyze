# exp/ad/clustering (v1.2)

## 🧭 目的
広告特徴量でのクラスタリングを実施。

## 🔄 実施内容
- 特徴量に以下を使用
カテゴリカル変数（ワンホットエンコーディング）："ad_platform","ad_type","target_gender","target_age_group","target_interests"
標準化："duration_days","total_budget"
- ad_id毎の平均CTR、CVRを特徴量として追加
- カテゴリ数の多い"target_interests"を2次元でPCAを行ってからクラスタリングを実施
- DBI、CHを評価指標として使用
- k=6 で各クラスタの特徴量の傾向と、平均CTR・CVR・CTVRを算出

## ⚙️ 使用データ
- ads.csv
- campaigns.csv
- events.csv

## 📊 現状の結果
- DBI：2.248 ⇒ 2.107、CH：42861.458 ⇒ 34300.271 でVe1.1よりもDBIは改善。
- 散布図でのまとまりはまだ見られない。
- 各クラスタの特徴は以下の通り
- クラスタごとのCTR、CVR差はみられるようになった。
- CTRへの影響：
    広告タイプとしてStories比重が大きいとCTRが低い
    男性向け比率が低いとCTRが低い（逆にCTRが高いクラスタは男性比率が高い）
    興味関心としてはgaming比重が高いとCTR低い、逆にfitness、sportsはCTRが高い
- CVRへの影響：
    Facebook比重が高い方がCVRは高い、また広告タイプはStoriesが多い方がCVRは高め、image比重が大きいとCVR低め
    18-24歳向けが大きい方がCVRは高め。
    興味関心はばらつきがあるが明確な傾向は見えない。


## 🚧 今後の課題
### 処理内容
- まとまりは不十分だが、クラスタごとの傾向は見えてきたのでkの値を2~10の範囲で検証

## 📝 備考
- branch: `feature/clustering-user-v3.1-draft`
- status: **検証中**
- 最終マージ予定：10/17 ver3