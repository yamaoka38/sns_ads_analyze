########################################################
# 0. 事前準備（データの読み込みと確認）
########################################################
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
# 1. クラスタリング
########################################################
# ============================================
# 1-1. クラスタリング実施
# ============================================
# df_dropをarrayに変換
X_train_arr = df_drop.to_numpy()

# kを3に設定
k = 3
print(f"k:{k}")

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)


# %% クラスタリングの評価
rng = np.random.default_rng(42)
silhouette = silhouette_score(X_train_arr, Y_km)
dbi = davies_bouldin_score(X_train_arr, Y_km)
ch = calinski_harabasz_score(X_train_arr, Y_km)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {dbi:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}")

# ============================================
# 1-2. 2次元散布図で可視化
# ============================================
# %% PCAで特徴量を2次元に圧縮
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_train_arr)

# セントロイド（クラスタ中心）もPCA空間に変換
centers_pca = pca.transform(km.cluster_centers_)

# 散布図で可視化
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_km, cmap="tab10", alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='red', s=200, marker='X', label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clusters_User_v1_ (PCA 2D Projection)')
plt.legend()
plt.savefig(f'../outputs/push/figures/kmeans_clustering_user_id_k={k}_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 1-3. ユーザーIDとクラスタIDのデータフレーム作成
# ============================================
cluster_df = pd.DataFrame({
    "user_id": df["user_id"].values,
    "cluster_id": Y_km
})

print(cluster_df.head())
print(cluster_df.shape)

########################################################
# 2. クラスタリング結果をBigQueryに送信
########################################################

# 送信したいBigQueryテーブル
push_table_id = "sns-ads-analyze.user_id_cluster.cluster_id_user"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE"   # ←既存テーブルを上書き
)

job = client.load_table_from_dataframe(cluster_df, push_table_id, job_config=job_config)
job.result()  # 完了待ち

print("BigQuery に書き込みました！")