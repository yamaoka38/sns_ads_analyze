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


# カテゴリカル変数をワンホットエンコーディング
cat_cols = ['ad_platform','ad_type','target_gender','target_age_group']
a = pd.get_dummies(a, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{a.head(10)}')

a_encoded = a.copy()


# 数値を標準化
scaler = StandardScaler()
num_cols = ["imp_cnt","click_cnt","purch_cnt","eng_cnt"]
a_encoded[num_cols] = scaler.fit_transform(a_encoded[num_cols])

a_encoded = a_encoded.drop(columns=["ad_id"])
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

# kを6に設定
k = cfg["clustering"]["ad_k"]
print(f"k:{k}")

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)

# クラスタリングの評価
rng = np.random.default_rng(42)
#sample_idx = rng.choice(len(X_train_arr), 10000, replace=False) #シルエットスコアの対象データをランダムに抽出
#silhouette = silhouette_score(X_train_arr[sample_idx], Y_km[sample_idx])
silhouette = silhouette_score(X_train_arr, Y_km)
dbi = davies_bouldin_score(X_train_arr, Y_km)
ch = calinski_harabasz_score(X_train_arr, Y_km)

print(f"Silhouette Score: {silhouette:.3f}")
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
plt.savefig(f'../outputs/push/figures/kmeans_clustering_adid_k={k}_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

'''
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
df_cluster_feat.to_csv(f"../outputs/push/df_clustering_ad_feat_k={k}_{timestamp}.csv")

# %% -- ヒートマップ画像作成
# 数値を正規化
df_scaled = df_cluster_feat.copy()
scaler = MinMaxScaler()
df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled)

# dfを分割（可視化しやすくする）
cols1 = [
    "duration_days",
    "total_budget",
    "ad_platform_Facebook",
    "ad_platform_Instagram",
    "ad_type_Carousel",
    "ad_type_Image",
    "ad_type_Stories",
    "ad_type_Video",
    "target_gender_All",
    "target_gender_Female",
    "target_gender_Male",
    "target_age_group_18-24",
    "target_age_group_25-34",
    "target_age_group_35-44",
    "target_age_group_All",
    "impressions",
    "clicks",
    "conversions",
    "CTR",
    "CVR",
    "CTVR"
]

cols2 = [
    "art",
    "fashion",
    "finance",
    "fitness",
    "food",
    "gaming",
    "health",
    "lifestyle",
    "news",
    "photography",
    "sports",
    "technology",
    "travel",
    "impressions",
    "clicks",
    "conversions",
    "CTR",
    "CVR",
    "CTVR"
]

df_cluster_feat1 = df_cluster_feat[cols1]
df_cluster_feat2 = df_cluster_feat[cols2]
df_scaled1 = df_scaled[cols1]
df_scaled2 = df_scaled[cols2]


# ヒートマップ描画
fig,axes = plt.subplots(2,1,figsize=(12,10))

sns.heatmap(
    df_scaled1,
    cmap="coolwarm",
    annot=df_cluster_feat1.round(2),
    fmt="",
    linewidth=0.5,
    cbar=True,
    ax=axes[0]
)
axes[0].set_title("ad_cluster_feat(1)",fontsize=14)

sns.heatmap(
    df_scaled2,
    cmap="coolwarm",
    annot=df_cluster_feat2.round(2),
    fmt="",
    linewidth=0.5,
    cbar=True,
    ax=axes[1]
)
axes[1].set_title("ad_cluster_feat(2)",fontsize=14)


plt.tight_layout()

# ④ 画像として保存
plt.savefig(f"../outputs/push/figures/cluster_feat_heatmap_ad_{timestamp}.png", dpi=150)
plt.show()


# ============================================
# 2-4. 広告クラスタIDを元データに結合
# ============================================
# %% クラスタ番号をtrain_allに結合
Y_km_s = pd.Series(Y_km, index=train_all.index, name="ad_cluster_id")
df_add_id = train_all.join(Y_km_s)
df_add_id.to_csv(f"../outputs/df_train_all_user-ad_cluster_id_{timestamp}.csv")

'''