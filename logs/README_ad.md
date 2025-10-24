# exp/ad/clustering (v1.5)

## 🧭 目的
広告特徴量でのクラスタリングを実施。

## 🔄 実施内容
- 特徴量に以下を使用
カテゴリカル変数（ワンホットエンコーディング）："ad_platform","ad_type","target_gender","target_age_group","target_interests"
標準化："duration_days","total_budget"
- ad_id毎の平均CTR、CVRを特徴量として追加
- カテゴリ数の多い"target_interests"を2次元でPCAを行ってからクラスタリングを実施
- DBI、CHを評価指標として使用
- k=5，6 でそれぞれクラスタリングした散布図と特徴量の可視化を確認し、最適なクラスタ数を最終判断

## ⚙️ 使用データ
- ads.csv
- campaigns.csv
- events.csv

## 📊 現状の結果
- kの増加に伴い、DBI,CHともに低下
- 詳細スコアは以下の通り
k	DBI	CH
5	2.1280781785063754	36976.22651
6	2.106792925112613	34300.27086714718


- 散布図、特徴量を総合的に判断し、k=5 を採用
（似た傾向が見られるが、k=5の方がより差分がわかりやすく、クラスタ数も少ないため説明性が高い）

- CTRの傾向
    配信面 InstagramがCTR高め
    興味関心 fitness、Spots、financeが高め。逆にgamingは低め
- CVRの傾向（CTVRもCTRより相関性が強く、こちらがPurchaseに大きく影響）
    広告タイプ Storiesが高め
    興味関心 photography、technology、newsが高め。artは低め

⇒InstagramのStoriesを中心に展開すると効果がよさそう。
写真やテクノロジー（カメラ・ガジェット好きユーザーに予算を寄せつつ、フィットネスやスポーツなどアクティブなイメージを持たせた広告でアプローチすると効果が最大化できそう。

## 🚧 今後の課題
### 処理内容
- ユーザー特徴量、広告特徴量をマージして再度クラスタリング
    - k=2~10 でまずは比較
    - 良さそうなkをいくつか絞り込み、散布図・特徴量を比較


## 📝 備考
- branch: `feature/clustering-user-v3.1-draft`
- status: **検証中**
- 最終マージ予定：10/17 ver3