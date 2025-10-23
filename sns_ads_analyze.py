########################################################
#  事前準備（データの読み込みと確認）
########################################################

# %% 必要なモジュールのインポート
from tkinter.constants import X
from datetime import datetime
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

# %%  データセットの読み込み
df_events = pd.read_csv('rawdata/ad_events.csv')
df_ads = pd.read_csv('rawdata/ads.csv')
df_cps = pd.read_csv('rawdata/campaigns.csv')
df_users = pd.read_csv('rawdata/users.csv')

'''
# 読み込みデータの確認
print(f'df_events:{df_events.head(5)}')
print(f'df_ads:{df_ads.head(5)}')
print(f'df_cps:{df_cps.head(5)}')
print(f'df_users:{df_users.head(5)}')
'''
# %% # データの中身を確認する関数を定義
def check_data(df):
    print('上位10件')
    print(df.head(10))
    print()
    print('データの形状')
    print(df.shape)
    print('データ型')
    print(df.dtypes)
    print()
    print('基本統計量(数値)')
    print(df.describe())
    print('基本統計量(カテゴリカル変数)')
    print(df.describe(exclude='number'))
    print()

# %% 全データをマージ
df_merged = pd.merge(df_events, df_ads, on='ad_id', how='left')
df_merged = pd.merge(df_merged, df_cps, on='campaign_id', how='left')
df_merged = pd.merge(df_merged, df_users, on='user_id', how='left')

# %% event_typeをワンホットエンコーディングで各カラムに変換
df_merged = pd.get_dummies(df_merged,columns=['event_type'],dtype=int)

########################################################
# 前処理（分割前）
########################################################

# 目的変数の処理
# %% imp列を追加（すべての値を1にする）
df_merged["imp"] = 1
# %% click列を追加（Purchase or event_type_clickが1の時、1を入れる。その他は0）
df_merged["click"] = np.where((df_merged["event_type_Click"] ==1) |(df_merged["event_type_Purchase"] ==1), 1, 0)
# %% Purchaseの列名を変更
df_merged = df_merged.rename(columns={"event_type_Purchase":"Purchase"})
# %% Engagement列を追加
df_merged["engagement"] = np.where((df_merged["event_type_Comment"]==1) |(df_merged["event_type_Like"]==1) |(df_merged["event_type_Share"]==1), 1, 0)


# %% timestampから月・日・開始日からの経過日数カラムと時間カラムを作成
df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"]) # timestampをdatetime型に変換
df_merged["month"] = df_merged["timestamp"].dt.month
df_merged["day"] = df_merged["timestamp"].dt.day
df_merged["day_from_start"] = (df_merged["timestamp"] - df_merged["timestamp"].min()) .dt.days
df_merged["hour"] = df_merged["timestamp"].dt.hour

## hourについて23時と0時を遠いと判断させないために、周期性を持たせる
df_merged["hour_sin"] = np.sin(2*np.pi*df_merged["hour"]/24)
df_merged["hour_cos"] = np.cos(2*np.pi*df_merged["hour"]/24)

# %% interestを変換
## カンマ区切りをリストに変換
df_merged["interests_list"] = df_merged["interests"].str.split(",")
## リストをワンホットエンコーディング
df_interests = df_merged["interests_list"].explode().str.strip().str.get_dummies().groupby(level=0).sum()
## 元のdfに結合
df_merged = pd.concat([df_merged, df_interests], axis=1)

# %% データを学習データとテストデータに分割
## まずは目的変数を設定せずに、Clickの比率を維持したまま分割
train_idx, test_idx = train_test_split(df_merged.index,test_size=0.2, random_state=0, stratify=df_merged["click"])

train_all = df_merged.loc[train_idx].reset_index(drop=True)
test_all = df_merged.loc[test_idx].reset_index(drop=True)


########################################################
# ユーザー指標でクラスタリング
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
'''
# 処理結果確認
print(f"ad_stats:{ad_stats.head(10)}")
print(f"X_train：{X_train.head(10)}")
print(f"test_all:{test_all.head(10)}")
ad_stats.head(30).to_csv(f"outputs/ad_stats_ctrcvr_head30_{timestamp}.csv")
X_train.head(30).to_csv(f"outputs/X_train_ctrcvr_head30_{timestamp}.csv")
test_all.head(30).to_csv(f"outputs/test_all_ctrcvr_head30_{timestamp}.csv")
'''
#%%
#print(X_train.head(5))
#X_train.head(30).to_csv(f"outputs/X_train_adfeat_{timestamp}.csv")

#%% target_interestsのユニーク値を確認
#print(X_train["target_interests"].unique())
#X_train["target_interests"].to_csv(f"outputs/X_train_tinterests_{timestamp}.csv")

#%% target_interests を変換
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
X_train.head(30).to_csv(f"outputs/X_train_pca_head30_{timestamp}.csv")

#%%


# %% カテゴリカル変数をワンホットエンコーディング
cat_cols = ["ad_platform","ad_type","target_gender","target_age_group"]
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{X_train_encoded.head(10)}')
#X_train_encoded.head(30).to_csv(f"outputs/X_train_adfeat_caten_{timestamp}.csv")

# %% imp/click/purchaseを削除
X_train_encoded_drop = X_train_encoded.copy()
X_train_encoded_drop = X_train_encoded_drop.drop(["ad_id","Purchase","imp","click","art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"], axis=1)
print(f"drop前:{X_train_encoded.head(5)}")
print(f"drop後:{X_train_encoded_drop.head(5)}")

# %% 数値特徴量 を標準化
scaler = StandardScaler()
num_cols = ["duration_days","total_budget","avg_ctr","avg_cvr"]
X_train_encoded_drop[num_cols] = scaler.fit_transform(X_train_encoded_drop[num_cols])
print(f"標準化後：{X_train_encoded_drop.head(10)}")

#%%
# %% 前処理後のデータをCSVで確認
X_train_encoded_drop.head(1000).to_csv(f"outputs/X_train_encoded_drop_ad3_head_{timestamp}.csv")


# %% 訓練データをdfからarrayに変換
X_train_arr = X_train_encoded_drop.to_numpy()

# kを6に設定
k = 6

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)

# %%
# PCAで特徴量を2次元に圧縮
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_train_arr)

# セントロイド（クラスタ中心）もPCA空間に変換
centers_pca = pca.transform(km.cluster_centers_)

# 散布図で可視化
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_km, cmap='viridis', alpha=0.6)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='red', s=200, marker='X', label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-means Clusters_User_v1_ (PCA 2D Projection)')
plt.legend()
plt.savefig(f'outputs/figures/kmeans_clusters_ad3_k=6_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% クラスタリングの評価
# silhouette = silhouette_score(X_train_arr, Y_km) #シルエットスコアは計算が重いので割愛
dbi = davies_bouldin_score(X_train_arr, Y_km)
ch = calinski_harabasz_score(X_train_arr, Y_km)

# print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {dbi:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}")

# %% クラスタ番号をdfに結合
Y_km_s = pd.Series(Y_km, index=X_train_encoded.index, name="cluster")
df = X_train_encoded.join(Y_km_s)
df.head(10).to_csv(f"outputs/df_cluster_k=6_ad3_{timestamp}.csv")

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

# %% 各クラスタの特徴量傾向を把握(数値の平均値)
cluster_summary = df.assign(cluster=Y_km).groupby("cluster").mean(numeric_only=True)
print(cluster_summary)

# %% 全出力をマージ
df_cluster_feat = pd.concat([cluster_summary,agg],axis=1)
print(df_cluster_feat)
df_cluster_feat.to_csv(f"outputs/df_cluster_ad3_feat_k=6_{timestamp}.csv")

