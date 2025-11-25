########################################################
# 0. 事前準備（データの読み込みと確認）
########################################################

# %% 必要なモジュールのインポート
import yaml
from tkinter.constants import X
from datetime import datetime
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# %%  タイムスタンプを取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(timestamp)

# %% yamlを読み込み
with open("../hub/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# %% --- 学習データを読み込み
# 出力フォルダを指定
output_dir = Path("../outputs")

# ファイル名のパターン（例：user_clusters_20251024_1530.csv）
pattern = re.compile(r"df_train_all_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files = sorted([f for f in output_dir.glob("df_train_all_*.csv") if pattern.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files:
    raise FileNotFoundError("df_train_all_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

# 最新ファイルを選択（ファイル名昇順の最後＝最新）
latest_file = files[-1]
print(f"読み込み対象ファイル: {latest_file.name}")

# CSVを読み込み
train_all = pd.read_csv(latest_file)

########################################################
# 1. 前処理
########################################################

# ============================================
# 1-1. 集計前の処理
# ============================================

# %% 曜日を周期エンコーディング
## 曜日を数値化(マッピング)
weekday_map = {
        "Monday":0,
        "Tuesday":1,
        "Wednesday":2,
        "Thursday":3,
        "Friday":4,
        "Saturday":5,
        "Sunday":6,
    }

# 数値化
train_all["weekday_num"] = train_all["day_of_week"].map(weekday_map)

train_all["weekday_sin"] = np.sin(2 * np.pi * train_all["weekday_num"] / 7)
train_all["weekday_cos"] = np.cos(2 * np.pi * train_all["weekday_num"] / 7)
train_all = train_all.drop(columns=["weekday_num","day_of_week"])

print(train_all.head(5))


# ============================================
# 1-2. ユーザーIDで集計したテーブル作成
# ============================================




# ユーザー集約テーブル作成（学習期間のみで作る）
u = train_all.groupby('user_id').agg(
    imp_cnt=('imp', 'sum'),
    click_cnt=('click', 'sum'),
    purch_cnt=('Purchase', 'sum'),
    eng_cnt=('engagement', 'sum'),
    avg_hour=('hour', 'mean'),
    avg_weekday_sin=('weekday_sin', 'mean'),
    avg_weekday_cos=('weekday_cos', 'mean'),
    
).reset_index()

# 比率指標を追加
u['ctr'] = (u['click_cnt'] / u['imp_cnt'].clip(lower=1)).fillna(0)
u['cvr'] = (u['purch_cnt'] / u['click_cnt'].clip(lower=1)).fillna(0)
u['eng_rate'] = (u['eng_cnt'] / u['imp_cnt'].clip(lower=1)).fillna(0)

# 属性情報を結合
# 興味関心一覧（割愛）　    "art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"
user_attrs = train_all[
    ['user_id','user_age','user_gender',]
    ].drop_duplicates('user_id')
print("user_attrs.head()")
print(user_attrs.head())
u = u.merge(user_attrs, on='user_id', how='left')
print("u.head()")
print(u.head())

# ============================================
# 1-3. 特徴量変換
# ============================================

# カテゴリカル変数をワンホットエンコーディング
cat_cols = ["user_gender"]
u_encoded = pd.get_dummies(u, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{u_encoded.head(10)}')

# 数値を標準化
scaler = StandardScaler()
# num_cols = ["user_age","imp_cnt","click_cnt","purch_cnt","eng_cnt","avg_hour"]
num_cols = ["user_age","avg_hour"]
u_encoded[num_cols] = scaler.fit_transform(u_encoded[num_cols])
print(f'標準化結果：{u_encoded.head(10)}')
#後処理用に、下記も計算
mean_age = u["user_age"].mean()
std_age = u["user_age"].std()
print(f'平均年齢：{mean_age}')
print(f'年齢標準偏差：{std_age}')

print(u_encoded.head(5))

u_encoded_drop = u_encoded.drop(columns=["user_id"])


########################################################
# 2. クラスタリング（最適なkを検証）
########################################################
# ============================================
# 2-1. クラスタリング実施
# ============================================
# %% 訓練データをdfからarrayに変換
X_train_arr = u_encoded_drop.to_numpy()

# まずはkの検証範囲を設定
k_range = range(2, 11)

# スコア格納リストを作成
sil_list, dbi_list, ch_list =[],  [], []
for k in k_range:
    # kmインスタンスを作成
    km = KMeans(n_clusters=k, init= "random", random_state=0, n_init="auto")
    # モデルの学習と予測を実行
    Y_km = km.fit_predict(X_train_arr)
    print(Y_km)

    # クラスタリングの評価
    sil = silhouette_score(X_train_arr, Y_km)
    dbi = davies_bouldin_score(X_train_arr, Y_km)
    ch = calinski_harabasz_score(X_train_arr, Y_km)

    sil_list.append(sil)
    dbi_list.append(dbi)
    ch_list.append(ch)

    print(f"k={k} | SIL={sil:.3f}, DBI={dbi:.3f}, CH={ch:.1f}")

# スコアをデータフレーム化
score_df = pd.DataFrame({
    "k": k_range,
    "SIL": sil_list,
    "DBI": dbi_list,
    "CH": ch_list
})

score_df.to_csv(f"../outputs/push/figures/clustering_user_score_compare-k_{timestamp}.csv")

# ベストスコアを確認
best_k_sil = score_df.loc[score_df["SIL"].idxmax(), "k"]
best_k_dbi = score_df.loc[score_df["DBI"].idxmin(), "k"]
best_k_ch = score_df.loc[score_df["CH"].idxmax(), "k"]
print(f"・SIL最大 → SIL={score_df["SIL"].max()}, k={best_k_sil}")
print(f"・DBI最小 → DBI={score_df["DBI"].min()}, k={best_k_dbi}")
print(f"・CH最大 → CH={score_df["CH"].max()}, k={best_k_ch}")

# --- グラフ描画 ---
fig, ax1 = plt.subplots(figsize=(8, 5))
color_ch = "tab:blue"
color_dbi = "tab:red"
color_sil = "tab:green"

# CH（左軸）
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("CH (↑)", color=color_ch)
ax1.plot(k_range, ch_list, marker="s", color=color_ch, label="CH (↑)")
ax1.tick_params(axis="y", labelcolor=color_ch)

# SIL & DBI（右軸）
ax2 = ax1.twinx()  # 右軸を作成
ax2.set_ylabel("Silhouette (↑) /DBI (↓)", color="grey")
ax2.plot(k_range, sil_list, marker="D", color=color_sil, label="Silhouette (↑) ")
ax2.plot(k_range, dbi_list, marker="o", color=color_dbi, label="DBI (↓)")
ax2.tick_params(axis="y")

# 凡例の統合
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# グラフタイトルと凡例
fig.suptitle("SIL (↑) & DBI (↓) & CH (↑) by Cluster Number", fontsize=13)
fig.tight_layout()
plt.savefig(f"../outputs/push/figures/kmeans_clusters_user_compare_k_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.show()


