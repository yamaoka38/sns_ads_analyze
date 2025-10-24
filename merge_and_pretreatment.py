########################################################
#  事前準備（データの読み込みと確認）
########################################################

# %% 必要なモジュールのインポート
from tkinter.constants import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
check_data(df_merged)

# %% event_typeをワンホットエンコーディングで各カラムに変換
df_merged = pd.get_dummies(df_merged,columns=['event_type'],dtype=int)

########################################################
# 前処理（分割前）
########################################################

# 目的変数の処理
# %% imp列を追加（すべての値を1にする）
df_merged["imp"] = 1
print(df_merged[["event_type_Impression","imp"]].head(50) )
print(df_merged[["event_type_Impression","imp"]].describe(),"\n")

# %% click列を追加（Purchase or event_type_clickが1の時、1を入れる。その他は0）
df_merged["click"] = np.where((df_merged["event_type_Click"] ==1) |(df_merged["event_type_Purchase"] ==1), 1, 0)
print(df_merged[["event_type_Click","event_type_Purchase","click"]].head(30),"\n")
print(df_merged[["event_type_Click","event_type_Purchase","click"]].describe(),"\n")
print(df_merged[["imp","event_type_Click","event_type_Purchase","click"]].sum(),"\n")

# %% Purchaseの列名を変更
df_merged = df_merged.rename(columns={"event_type_Purchase":"Purchase"})

# %% Engagement列を追加
df_merged["engagement"] = np.where((df_merged["event_type_Comment"]==1) |(df_merged["event_type_Like"]==1) |(df_merged["event_type_Share"]==1), 1, 0)
print(df_merged[["event_type_Comment","event_type_Like","event_type_Share","engagement"]].head(30),"\n")
print(df_merged[["event_type_Comment","event_type_Like","event_type_Share","engagement"]].describe(),"\n")
print(df_merged[["event_type_Comment","event_type_Like","event_type_Share","engagement"]].sum(),"\n")

# %% CSVでdfを確認
df_merged.head(100).to_csv('outputs/df_merged_event_pre_head100.csv')

# %% timestampから月・日・開始日からの経過日数カラムと時間カラムを作成
df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"]) # timestampをdatetime型に変換
df_merged["month"] = df_merged["timestamp"].dt.month
df_merged["day"] = df_merged["timestamp"].dt.day
df_merged["day_from_start"] = (df_merged["timestamp"] - df_merged["timestamp"].min()) .dt.days
df_merged["hour"] = df_merged["timestamp"].dt.hour

## hourについて23時と0時を遠いと判断させないために、周期性を持たせる
df_merged["hour_sin"] = np.sin(2*np.pi*df_merged["hour"]/24)
df_merged["hour_cos"] = np.cos(2*np.pi*df_merged["hour"]/24)

# 処理結果を確認
print(df_merged[["timestamp","month","day","day_from_start","hour","hour_sin","hour_cos"]].dtypes,"\n")
print(df_merged[["timestamp","month","day","day_from_start","hour","hour_sin","hour_cos"]].head(10),"\n")


# %% interestを変換
## カンマ区切りをリストに変換
df_merged["interests_list"] = df_merged["interests"].str.split(",")
## print(df_merged["interests_list"].head(10))
print(df_merged["interests_list"].explode().head(10))
## リストをワンホットエンコーディング
df_interests = df_merged["interests_list"].explode().str.strip().str.get_dummies().groupby(level=0).sum()
print(df_interests.head(10))
## 元のdfに結合
df_merged = pd.concat([df_merged, df_interests], axis=1)
print(df_merged.head(10))

# %% CSVでdfを確認
df_merged.head(100).to_csv('outputs/df_merged_feature_pre_head100.csv')


# %% データを学習データとテストデータに分割
## まずは目的変数を設定せずに、Clickの比率を維持したまま分割
train_idx, test_idx = train_test_split(df_merged.index,test_size=0.2, random_state=0, stratify=df_merged["click"])

train_all = df_merged.loc[train_idx].reset_index(drop=True)
test_all = df_merged.loc[test_idx].reset_index(drop=True)

# %% 
print('train_all')
check_data(train_all)

# %%
print('test_all')
check_data(test_all)
# %%
