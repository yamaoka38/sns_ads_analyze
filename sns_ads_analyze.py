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

# データの中身を確認
dataframes ={"df_events":df_events,
              "df_ads":df_ads,
              "df_cps":df_cps,
              "df_users":df_users}

for name,df in dataframes.items():
    print(name)
    check_data(df)

check_data(df_events)

# %% 全データをマージ
df_merged = pd.merge(df_events, df_ads, on='ad_id', how='left')
df_merged = pd.merge(df_merged, df_cps, on='campaign_id', how='left')
df_merged = pd.merge(df_merged, df_users, on='user_id', how='left')
check_data(df_merged)

# %% # interestの種類と件数を確認
print(f'interestsの件数：{df_merged['interests'].nunique()}')
print(df_merged['interests'].unique())

# %% event_typeをワンホットエンコーディングで各カラムに変換
df_merged = pd.get_dummies(df_merged,columns=['event_type'],dtype=int)

# %% 性別の分布確認
sns.countplot(x='user_gender',hue='event_type_Click',data=df_merged)
plt.show()

# %% 年齢の分布確認
fig = sns.FacetGrid(df_merged,col='event_type_Click',hue='event_type_Click',height=4)
fig.map(sns.histplot,'user_age',bins=100,kde=False)
plt.show()

# %% 曜日の分布確認
sns.countplot(x='day_of_week',hue='event_type_Click',data=df_merged)
plt.show()

# %% 国の分布確認
sns.countplot(x='country',hue='event_type_Click',data=df_merged)
plt.show()

# %% 広告タイプの分布確認
sns.countplot(x='ad_type',hue='event_type_Click',data=df_merged)
plt.show()

# %% 広告プラットフォームの分布確認
sns.countplot(x='ad_platform',hue='event_type_Click',data=df_merged)
plt.show()

# %%
