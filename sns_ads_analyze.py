########################################################
#  事前準備（データの読み込みと確認）
########################################################

# %% 必要なモジュールのインポート
from tkinter.constants import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt

# %%  データセットの読み込み
df_events = pd.read_csv('rawdata/ad_events.csv')
df_ads = pd.read_csv('rawdata/ads.csv')
df_cps = pd.read_csv('rawdata/campaigns.csv')
df_users = pd.read_csv('rawdata/users.csv')

# 読み込みデータの確認
print(f'df_events:{df_events.head(5)}')
print(f'df_ads:{df_ads.head(5)}')
print(f'df_cps:{df_cps.head(5)}')
print(f'df_users:{df_users.head(5)}')

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
"day_of_week","user_gender","user_age","hour_sin","hour_cos",
"art","fashion","finance","fitness","food","gaming","health","lifestyle","news","photography","sports","technology","travel"
]
X_train = train_all[use_cols]
X_train = X_train.copy()

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

# カテゴリカル変数（性別）をワンホットエンコーディング
cat_cols = ["user_gender"]
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=False, dtype=int)
print(f'ワンホットエンコーディング結果：{X_train_encoded.head(10)}')

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

# %% データをCSVで確認
X_train_encoded.head(1000).to_csv("outputs/X_train_encoded_user_2_head1000.csv")


# %% 訓練データをdfからarrayに変換
X_train_arr = X_train_encoded.to_numpy()

# まずはkを仮置き
k = 5

# kmインスタンスを作成
km = KMeans(n_clusters=k, init= "random", random_state=0, n_init='auto')
# モデルの学習と予測を実行
Y_km = km.fit_predict(X_train_arr)
print(Y_km)


# %% PCAで特徴量を2次元に圧縮
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

plt.savefig('outputs/figures/kmeans_clusters_user_v2.png', dpi=300, bbox_inches='tight')
plt.show()

# %% クラスタリングの評価
# silhouette = silhouette_score(X_train_arr, Y_km) #シルエットスコアは計算が重いので割愛
dbi = davies_bouldin_score(X_train_arr, Y_km)
ch = calinski_harabasz_score(X_train_arr, Y_km)

# print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {dbi:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}")

# %% X_train_arrをデータフレームに変換
df = pd.DataFrame(X_train_arr,columns=X_train_encoded.columns)
df["cluster"] = Y_km
df.head(10).to_csv("outputs/df_cluster2.csv")

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

# %% 各クラスタの特徴量傾向を把握(数値の平均値)
cluster_summary = df.assign(cluster=Y_km).groupby("cluster").mean(numeric_only=True)
print(cluster_summary)
cluster_summary.to_csv("outputs/df_cluster2_mean.csv")

# %% 各クラスタの特徴量傾向を把握(曜日)
# 件数集計
ct = pd.crosstab(df["cluster"], df["weekday"])
# 件数→割合（行方向に正規化）
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
# CSV出力
ct_pct.to_csv("outputs/cluster_weekday_ratio_2.csv", encoding="utf-8-sig")

# %%
