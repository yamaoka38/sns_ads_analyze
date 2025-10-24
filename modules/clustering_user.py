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

# %%  クラスタリングに使用するカラムを選択
use_cols = [
"day_of_week","user_gender","user_age","hour_sin","hour_cos","Purchase","imp","click"
]
X_train_s = train_all[use_cols]
X_train = X_train_s.copy()

#%%
print(X_train.head(5))


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
X_train["weekday_num"] = X_train["day_of_week"].map(weekday_map)

X_train["weekday_sin"] = np.sin(2 * np.pi * X_train["weekday_num"] / 7)
X_train["weekday_cos"] = np.cos(2 * np.pi * X_train["weekday_num"] / 7)
X_train = X_train.drop(columns=["weekday_num","day_of_week"])

print(X_train.head(5))

# %% カテゴリカル変数（性別）をワンホットエンコーディング
cat_cols = ["user_gender"]
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{X_train_encoded.head(10)}')

print(X_train_encoded.head(5))

# %% user_age を標準化
scaler = StandardScaler()
num_cols = ["user_age"]
X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
print(f'標準化結果：{X_train_encoded.head(10)}')
#後処理用に、下記も計算
mean_age = X_train["user_age"].mean()
std_age = X_train["user_age"].std()
print(f'平均年齢：{mean_age}')
print(f'年齢標準偏差：{std_age}')

print(X_train_encoded.head(5))

# %% 前処理データをCSVで確認
X_train_encoded.head(100).to_csv(f"../outputs/df_X_train_encoded_user_pretreatment_head100_{timestamp}.csv")

X_train_encoded_drop = X_train_encoded.copy()
X_train_encoded_drop = X_train_encoded_drop.drop(["Purchase","imp","click"], axis=1)

print(X_train_encoded.head(5))
print(X_train_encoded_drop.head(5))

########################################################
# 2. クラスタリング
########################################################
# ============================================
# 2-1. クラスタリング実施
# ============================================
# %% 訓練データをdfからarrayに変換
X_train_arr = X_train_encoded_drop.to_numpy()

# kを6に設定
k = cfg["clustering"]["user_k"]
print(f"k:{k}")

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)


# %% クラスタリングの評価
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
plt.title('K-means Clusters_User_v1_ (PCA 2D Projection)')
plt.legend()
plt.savefig(f'../outputs/push/figures/kmeans_clustering_user_k={k}_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2-3. クラスタごとの特徴量を可視化
# ============================================
# %% クラスタ番号をdf1に結合
Y_km_s = pd.Series(Y_km, index=X_train_encoded.index, name="cluster")
df = X_train_encoded.join(Y_km_s)
df.head(100).to_csv(f"../outputs/df_X_train_add_user_clustering_id_k={k}_head100_{timestamp}.csv")

# %% 年齢を元に戻す（手計算）
df["user_age"] = df["user_age"] * std_age + mean_age
print(df["user_age"].head(30))

# %% 時間を24時間表記に戻す
df["hour"] = np.degrees(np.arctan2(df["hour_sin"], df["hour_cos"])) * (24 / 360)
df["hour"] = (df["hour"] + 24) % 24
print(df["hour"].head(30))

# %% 曜日を元に戻す
df["weekday_num"] = np.degrees(np.arctan2(df["weekday_sin"], df["weekday_cos"])) * (7 / 360)
df["weekday_num"] = (df["weekday_num"] + 7) % 7  # 負の値補正
df["weekday_num"] = df["weekday_num"].round().astype(int)

# キーと値を入れ替えた辞書を作成（逆マップ）
weekday_map_inv = {v: k for k, v in weekday_map.items()}

df['weekday'] = df['weekday_num'].map(weekday_map_inv)
print(df['weekday'].head(30))

# %% 各クラスタの特徴量傾向を把握(曜日)
# 件数集計
ct = pd.crosstab(df["cluster"], df["weekday"])
# 件数→割合（行方向に正規化）
ct_pct = ct.div(ct.sum(axis=1), axis=0) 
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ct_pct = ct_pct[weekday_order]
print(ct_pct)

# %% クラスタ毎のCTR・CVR・CTVRを計算
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
#%%

# %% 各クラスタの特徴量傾向を把握(数値の平均値)
cluster_summary = df.assign(cluster=Y_km).groupby("cluster").mean(numeric_only=True)
print(cluster_summary)

# %% 全出力をマージ
df_cluster_feat = pd.concat([cluster_summary,ct_pct,agg],axis=1)
print(df_cluster_feat)

# %%  不要な列を削除
drop_list = ["hour_sin","hour_cos","Purchase","imp","click","weekday_sin","weekday_cos"]
df_cluster_feat_drop = df_cluster_feat.copy()
df_cluster_feat_drop = df_cluster_feat_drop.drop(columns=drop_list,axis=1)
print(df_cluster_feat_drop)
df_cluster_feat_drop.to_csv(f"../outputs/push/df_clustering_user_feat_k={k}_{timestamp}.csv")
# %%

# ============================================
# 2-4. ユーザークラスタIDを元データに結合
# ============================================
# %% クラスタ番号をtrain_allに結合
Y_km_s = pd.Series(Y_km, index=train_all.index, name="user_cluster_id")
df_add_id = train_all.join(Y_km_s)
df_add_id.to_csv(f"../outputs/df_train_all_user_cluster_id_{timestamp}.csv")