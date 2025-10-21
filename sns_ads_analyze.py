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
"day_of_week","user_gender","user_age","hour_sin","hour_cos"
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
# X_train_encoded.head(1000).to_csv("outputs/X_train_encoded_user_3_head1000.csv")


# %% 訓練データをdfからarrayに変換
X_train_arr = X_train_encoded.to_numpy()

# まずはkの検証範囲を設定
k_range = range(2, 11)

# スコア格納リストを作成
dbi_list, ch_list = [], []

for k in k_range:
    # kmインスタンスを作成
    km = KMeans(n_clusters=k, init= "random", random_state=0, n_init="auto")
    # モデルの学習と予測を実行
    Y_km = km.fit_predict(X_train_arr)
    print(Y_km)

    # クラスタリングの評価
    # silhouette = silhouette_score(X_train_arr, Y_km) #シルエットスコアは計算が重いので割愛
    dbi = davies_bouldin_score(X_train_arr, Y_km)
    ch = calinski_harabasz_score(X_train_arr, Y_km)

    dbi_list.append(dbi)
    ch_list.append(ch)

    print(f"k={k} | DBI={dbi:.3f}, CH={ch:.1f}")

# スコアをデータフレーム化
score_df = pd.DataFrame({
    "k": k_range,
    "DBI": dbi_list,
    "CH": ch_list
})

score_df.to_csv("outputs/clustering__user_score_compare-k.csv")

# ベストスコアを確認
best_k_dbi = score_df.loc[score_df["DBI"].idxmin(), "k"]
best_k_ch = score_df.loc[score_df["CH"].idxmax(), "k"]
print(f"・DBI最小 → DBI={score_df["DBI"].min()}, k={best_k_dbi}")
print(f"・CH最大 → CH={score_df["CH"].max()}, k={best_k_ch}")

# --- グラフ描画 ---
fig, ax1 = plt.subplots(figsize=(8, 5))
color_ch = "tab:blue"
color_dbi = "tab:red"

# CH（左軸）
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("Calinski-Harabasz (↑)", color=color_ch)
ax1.plot(k_range, ch_list, marker="s", color=color_ch, label="CH (↑)")
ax1.tick_params(axis="y", labelcolor=color_ch)

# DBI（右軸）
ax2 = ax1.twinx()  # 右軸を作成
ax2.set_ylabel("Davies-Bouldin (↓)", color=color_dbi)
ax2.plot(k_range, dbi_list, marker="o", color=color_dbi, label="DBI (↓)")
ax2.tick_params(axis="y", labelcolor=color_dbi)

# グラフタイトルと凡例
fig.suptitle("DBI (↓) & CH (↑) by Cluster Number", fontsize=13)
fig.tight_layout()
plt.savefig("outputs/figures/kmeans_clusters_user_compare_k.png", dpi=300, bbox_inches="tight")
plt.show()
