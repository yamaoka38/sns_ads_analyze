########################################################
# 0. 事前準備（データの読み込みと確認）
########################################################
from google.cloud import bigquery
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt

# %%  タイムスタンプを取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(timestamp)

# ============================================
# 0-1. BigQueryからテーブル読み込み
# ============================================

# 認証情報（Service Account JSONのパスを指定）
client = bigquery.Client.from_service_account_json("../../keys/sns-ads-analyze.json")

# 読み込みたいBigQueryテーブル
table_id = "sns-ads-analyze.user_id_cluster.user_features_master"

# テーブル読み込み
query = f"SELECT user_id, click_cnt, purchase_cnt, ctr, cvr, avg_weekday_sin, avg_weekday_cos, avg_hour_sin, avg_hour_cos FROM `{table_id}`"
df = client.query(query).to_dataframe()

print(df.head())
print(df.shape)

# ============================================
# 0-2. 前処理
# ============================================
# -- 値の大きい特徴量を標準化
df_sca = df.copy()
scaler = StandardScaler()
sca_cols = ["click_cnt", "purchase_cnt"]
df_sca[sca_cols] = scaler.fit_transform(df[sca_cols])
print(df_sca.head())
df_drop = df_sca.drop(columns=["user_id"])
print(df_drop.head())


########################################################
# 1. クラスタリング（k数を比較）
########################################################
# ============================================
# 1-1. 各k数でのスコアを作成
# ============================================
# df_dropをarrayに変換
X_train_arr = df_drop.to_numpy()

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
print(f"・SIL最大 → SIL={score_df['SIL'].max()}, k={best_k_sil}")
print(f"・DBI最小 → DBI={score_df['DBI'].min()}, k={best_k_dbi}")
print(f"・CH最大 → CH={score_df['CH'].max()}, k={best_k_ch}")

# ============================================
# 1-2. グラフ作成
# ============================================
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
plt.savefig(f"../outputs/push/figures/kmeans_clusters_userid_compare_k_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.show()
