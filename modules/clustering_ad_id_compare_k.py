########################################################
#  0. 事前準備（データの読み込みと確認）
########################################################
# ============================================
# 0-1. 必要なモジュールのインポート
# ============================================
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

# ============================================
# 0-2. 学習データを読み込み
# ============================================
# 出力フォルダを指定
output_dir = Path("../outputs")

# ファイル名のパターン（例：user_clusters_20251024_1530.csv）
pattern = re.compile(r"df_train_all_userid_cluster_id_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files = sorted([f for f in output_dir.glob("df_train_all_userid_cluster_id*.csv") if pattern.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files:
    raise FileNotFoundError("df_train_all_userid_cluster_id_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

# 最新ファイルを選択（ファイル名昇順の最後＝最新）
latest_file = files[-1]
print(f"読み込み対象ファイル: {latest_file.name}")

# CSVを読み込み
train_all = pd.read_csv(latest_file)

# ============================================
# 0-3. テストデータを読み込み
# ============================================
# %% --- テストデータを読み込み
# ファイル名のパターン（例：user_clusters_20251024_1530.csv）
pattern_test = re.compile(r"df_test_all_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files_test = sorted([f for f in output_dir.glob("df_test_all_*.csv") if pattern_test.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files_test:
    raise FileNotFoundError("df_test_all_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

# 最新ファイルを選択（ファイル名昇順の最後＝最新）
latest_file_test = files_test[-1]
print(f"読み込み対象ファイル(テスト): {latest_file_test.name}")

# CSVを読み込み
test_all = pd.read_csv(latest_file_test)

########################################################
# 1. 前処理
########################################################
# ============================================
# 1-1. ユーザーIDで集計したテーブル作成
# ============================================

# 広告集約テーブル作成（学習期間のみで作る）
a = train_all.groupby('ad_id').agg(
    imp_cnt=('imp', 'sum'),
    click_cnt=('click', 'sum'),
    purch_cnt=('Purchase', 'sum'),
    eng_cnt=('engagement', 'sum'),
#    avg_hour=('hour', 'mean'),
#    avg_weekday_sin=('weekday_sin', 'mean'),
#    avg_weekday_cos=('weekday_cos', 'mean'),
    ).reset_index()

# 比率指標を追加
a['ctr'] = (a['click_cnt'] / a['imp_cnt'].clip(lower=1)).fillna(0)
a['cvr'] = (a['purch_cnt'] / a['click_cnt'].clip(lower=1)).fillna(0)
a['eng_rate'] = (a['eng_cnt'] / a['imp_cnt'].clip(lower=1)).fillna(0)

# 属性情報を結合
ad_attrs = train_all[
    ['ad_id','ad_platform','ad_type','target_gender','target_age_group','target_interests']
    ].drop_duplicates('ad_id')
print("ad_attrs.head()")
print(ad_attrs.head())
a = a.merge(ad_attrs, on='ad_id', how='left')
print("a.head()")
print(a.head())

# ============================================
# 1-2. 特徴量変換
# ============================================

# %% --- target_interestsを変換
a["t_interests_list"] = a["target_interests"].str.split(",")
## リストをワンホットエンコーディング
df_t_interests = a["t_interests_list"].explode().str.strip().str.get_dummies().groupby(level=0).sum()
print("df_t_interests")
print(df_t_interests.head(10))
## 元のdfに結合
a = pd.concat([a, df_t_interests], axis=1)
a = a.drop(["t_interests_list","target_interests"],axis=1)
print(a.head())

# PCAで次元圧縮
pca = PCA(n_components=2)
df_t_interests_pca = pca.fit_transform(df_t_interests)
a["pca_interest_1"] = df_t_interests_pca[:, 0]
a["pca_interest_2"] = df_t_interests_pca[:, 1]

# カテゴリカル変数をワンホットエンコーディング
cat_cols = ['ad_platform','ad_type','target_gender','target_age_group']
a = pd.get_dummies(a, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{a.head(10)}')

a_encoded = a.copy()


# 数値を標準化
scaler = StandardScaler()
num_cols = ["imp_cnt","click_cnt","purch_cnt","eng_cnt"]
a_encoded[num_cols] = scaler.fit_transform(a_encoded[num_cols])

a_encoded = a_encoded.drop(columns=["ad_id","art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"])
#a_encoded = a_encoded.drop(columns=num_cols)
print(a_encoded.head())


########################################################
# 2. クラスタリング
########################################################
# ============================================
# 2-1. クラスタリング実施
# ============================================

# %% 訓練データをdfからarrayに変換
X_train_arr = a_encoded.to_numpy()

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

score_df.to_csv(f"../outputs/push/figures/clustering_adid_score_compare-k_{timestamp}.csv")

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
plt.savefig(f"../outputs/push/figures/kmeans_clusters_adid_compare_k_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.show()
