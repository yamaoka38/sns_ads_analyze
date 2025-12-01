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
u = pd.get_dummies(u, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{u.head(10)}')

u_encoded = u.copy()
# 数値を標準化
scaler = StandardScaler()
num_cols = ["user_age","imp_cnt","click_cnt","purch_cnt","eng_cnt","avg_hour"]
#num_cols = ["user_age","avg_hour"]
u_encoded[num_cols] = scaler.fit_transform(u_encoded[num_cols])
print(f'標準化結果：{u_encoded.head(10)}')
#後処理用に、下記も計算
mean_age = u["user_age"].mean()
std_age = u["user_age"].std()
print(f'平均年齢：{mean_age}')
print(f'年齢標準偏差：{std_age}')

mean_hour = u["avg_hour"].mean()
std_hour = u["avg_hour"].std()
print(f'平均時間：{mean_hour}')
print(f'時間標準偏差：{std_hour}')

print(u_encoded.head(5))

u_encoded_drop = u_encoded.drop(columns=["user_id"])


########################################################
# 2. クラスタリング
########################################################
# ============================================
# 2-1. クラスタリング実施
# ============================================
# %% 訓練データをdfからarrayに変換
X_train_arr = u_encoded_drop.to_numpy()

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
plt.title('K-means Clusters_User_v1_ (PCA 2D Projection)')
plt.legend()
plt.savefig(f'../outputs/push/figures/kmeans_clustering_user_id_k={k}_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# 2-3. クラスタごとの特徴量を可視化
# ============================================
# %% クラスタ番号をdf1に結合
Y_km_s = pd.Series(Y_km, index=u.index, name="user_cluster_id")
df = u.join(Y_km_s)
df.head(100).to_csv(f"../outputs/df_X_train_add_userid_clustering_id_k={k}_head100_{timestamp}.csv")

# %% クラスタ毎のCTR・CVR・CTVRを計算
agg = (
    df
    .groupby("user_cluster_id", dropna=False)
    .agg(
        impressions=("imp_cnt", "sum"),
        clicks=("click_cnt", "sum"),
        conversions=("purch_cnt", "sum")
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
cluster_summary = df.assign(cluster=Y_km).groupby("user_cluster_id").mean(numeric_only=True)
print(cluster_summary)

# %% 全出力をマージ
df_cluster_feat = pd.concat([cluster_summary,agg],axis=1)
print(df_cluster_feat)

# %%  不要な列を削除
drop_list = ["imp_cnt","click_cnt","purch_cnt","eng_cnt","ctr","cvr"]
df_cluster_feat_drop = df_cluster_feat.copy()
df_cluster_feat_drop = df_cluster_feat_drop.drop(columns=drop_list,axis=1)
print(df_cluster_feat_drop)
df_cluster_feat_drop.to_csv(f"../outputs/push/df_clustering_userid_feat_k={k}_{timestamp}.csv")

# %% -- ヒートマップ画像作成
# 数値を正規化
df_scaled = df_cluster_feat_drop.copy()
scaler = MinMaxScaler()
df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled)

# ヒートマップ描画
plt.figure(figsize=(14,6))
sns.heatmap(
    df_scaled,
    cmap="coolwarm",
    annot=df_cluster_feat_drop.round(2),
    fmt="",
    linewidth=0.5,
    cbar=True
)
plt.title("user_cluster_feat",fontsize=14)
plt.tight_layout()

# ④ 画像として保存
plt.savefig(f"../outputs/push/figures/cluster_feat_heatmap_userid_{timestamp}.png", dpi=150)
plt.show()


# ============================================
# 2-4. ユーザークラスタIDを元データに結合
# ============================================
# %% クラスタ番号をtrain_allに結合
#Y_km_s = pd.Series(Y_km, index=train_all.index, name="user_cluster_id")
#df_add_id = train_all.join(Y_km_s)

df_add_cluster_id = train_all.merge(df[["user_id","user_cluster_id"]],on="user_id",how="left")
df[["user_id","user_cluster_id"]].to_csv(f"../outputs/userid_clusterid_list_{timestamp}.csv")
df_add_cluster_id.to_csv(f"../outputs/df_train_all_userid_cluster_id_{timestamp}.csv")
