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
from sklearn.preprocessing import StandardScaler
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
pattern = re.compile(r"df_train_all_user_cluster_id_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files = sorted([f for f in output_dir.glob("df_train_all_user_cluster_id*.csv") if pattern.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files:
    raise FileNotFoundError("df_train_all_user_cluster_id_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

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
# %%  クラスタリングに使用するカラムを選択
use_cols = [
"ad_id","ad_platform","ad_type",
"target_gender","target_age_group","target_interests","duration_days","total_budget",
"Purchase","imp","click"
]
X_train_s = train_all[use_cols]
X_train = X_train_s.copy()

print(X_train.head(5))

# --- 平均CTR・CVRの追加
# %% ad_id毎の平均CTR・CVRを計算
ad_stats = (
    X_train.groupby("ad_id")
    .agg(
        avg_ctr=("click", lambda x: x.sum() / X_train.loc[x.index, "imp"].sum()),
        avg_cvr=("Purchase", lambda x: x.sum() / X_train.loc[x.index, "click"].sum())
    )
    .reset_index()
)

X_train = X_train.merge(ad_stats, on="ad_id", how="left")

# ここで算出した平均CTR・CVRを特徴量としてテストデータにも反映
test_all = test_all.merge(ad_stats, on="ad_id", how="left")
# テストに存在する新しい ad_id は平均値で補完
test_all[["avg_ctr", "avg_cvr"]] = test_all[["avg_ctr", "avg_cvr"]].fillna(ad_stats[["avg_ctr", "avg_cvr"]].mean())
#%%

# %% --- target_interestsを変換
## カンマ区切りをリストに変換
X_train["t_interests_list"] = X_train["target_interests"].str.split(",")
## リストをワンホットエンコーディング
df_t_interests = X_train["t_interests_list"].explode().str.strip().str.get_dummies().groupby(level=0).sum()
print("df_t_interests")
print(df_t_interests.head(10))
## 元のdfに結合
X_train = pd.concat([X_train, df_t_interests], axis=1)
X_train = X_train.drop(["t_interests_list","target_interests"],axis=1)

# PCAで次元圧縮
pca = PCA(n_components=2)
df_t_interests_pca = pca.fit_transform(df_t_interests)
X_train["pca_interest_1"] = df_t_interests_pca[:, 0]
X_train["pca_interest_2"] = df_t_interests_pca[:, 1]

print("pca.explained_variance_ratio_")
print(pca.explained_variance_ratio_)

# %% --- カテゴリカル変数をワンホットエンコーディング
cat_cols = ["ad_platform","ad_type","target_gender","target_age_group"]
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{X_train_encoded.head(10)}')


# %% --- 不要な列を削除
X_train_encoded_drop = X_train_encoded.copy()
X_train_encoded_drop = X_train_encoded_drop.drop(["ad_id","Purchase","imp","click","art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"], axis=1)
print(f"drop前:{X_train_encoded.head(5)}")
print(f"drop後:{X_train_encoded_drop.head(5)}")

# %% --- 数値特徴量 を標準化
scaler = StandardScaler()
num_cols = ["duration_days","total_budget","avg_ctr","avg_cvr"]
X_train_encoded_drop[num_cols] = scaler.fit_transform(X_train_encoded_drop[num_cols])
print(f"標準化後：{X_train_encoded_drop.head(10)}")

# %% 前処理後のデータをCSVで確認
X_train_encoded_drop.head(100).to_csv(f"../outputs/df_X_train_encoded_ad_pretreatment_head100_{timestamp}.csv")

########################################################
# 2. クラスタリング
########################################################
# ============================================
# 2-1. クラスタリング実施
# ============================================

# %% 訓練データをdfからarrayに変換
X_train_arr = X_train_encoded_drop.to_numpy()

# kを6に設定
k = cfg["clustering"]["ad_k"]
print(f"k:{k}")

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)


# クラスタリングの評価
# silhouette = silhouette_score(X_train_arr, Y_km) #シルエットスコアは計算が重いので割愛
dbi = davies_bouldin_score(X_train_arr, Y_km)
ch = calinski_harabasz_score(X_train_arr, Y_km)

# print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {dbi:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}")

# ============================================
# 2-2. 2次元散布図で可視化
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
plt.title('K-means Clusters_ad_v1_ (PCA 2D Projection)')
plt.legend()
plt.savefig(f'../outputs/figures/kmeans_clustering_ad_k={k}_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2-3. クラスタごとの特徴量を可視化
# ============================================
# クラスタ番号をdfに結合
Y_km_s = pd.Series(Y_km, index=X_train_encoded.index, name="cluster")
df = X_train_encoded.join(Y_km_s)
df.head(100).to_csv(f"../outputs/df_X_train_add_ad_clustering_id_k={k}_head100_{timestamp}.csv")

# クラスタ毎のCTR・CVR・CTVRを計算
agg = (
    df
    .groupby("cluster", dropna=False)
    .agg(
        impressions=("imp", "sum"),
        clicks=("click", "sum"),
        conversions=("Purchase", "sum")
    )
)

agg["CTR"] = np.where(agg["impressions"] > 0,
                    agg["clicks"] / agg["impressions"], np.nan)
agg["CVR"] = np.where(agg["clicks"] > 0,
                    agg["conversions"] / agg["clicks"], np.nan)
agg["CTVR"] = np.where(agg["impressions"] > 0,
                    agg["CTR"] * agg["CVR"]*100, np.nan)
print(agg.reset_index())
print(agg)

# %% 各クラスタの特徴量傾向を把握(数値の平均値)
cluster_summary = df.assign(cluster=Y_km).groupby("cluster").mean(numeric_only=True)
print(cluster_summary)

# %% 全出力をマージ
df_cluster_feat = pd.concat([cluster_summary,agg],axis=1)
print(df_cluster_feat)
df_cluster_feat.to_csv(f"../outputs/df_clustering_ad_feat_k={k}_{timestamp}.csv")


# ============================================
# 2-4. 広告クラスタIDを元データに結合
# ============================================
# %% クラスタ番号をtrain_allに結合
Y_km_s = pd.Series(Y_km, index=train_all.index, name="ad_cluster_id")
df_add_id = train_all.join(Y_km_s)
df_add_id.to_csv(f"../outputs/df_train_all_user-ad_cluster_id_{timestamp}.csv")


