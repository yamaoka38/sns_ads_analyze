# exp/user/clustering (v1.0)

## 🧭 目的
ユーザー特徴量でのクラスタリングを実施。

## 🔄 実施内容
- 特徴量に以下を使用
"day_of_week","user_gender","user_age","hour_sin","hour_cos"
"art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"
- カテゴリカル変数として"day_of_week","user_gender"をワンホットエンコーディング
- k=5 を仮置きで設定
- 標準化＋PCAで2次元化して分布確認

## ⚙️ 使用データ
- users.csv
- events.csv

## 📊 現状の結果
- PCAのX軸は年齢の影響を強く受けている印象。（X軸のみでクラスタが決定されている状況）
- Y軸はカテゴリカル変数が多すぎて分散している。

## 🚧 今後の課題
- カテゴリカル変数のエンコーディング方法を見直して再検証
day_of_week：周期エンコーディング など

## 📝 備考
- branch: `feature/clustering-user-v3.1-draft`
- status: **検証中**
- 最終マージ予定：10/17 ver3