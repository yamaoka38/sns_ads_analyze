########################################################
# 0. 事前準備（データの読み込みと確認）
########################################################

# %% 必要なモジュールのインポート
import yaml
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# %%  タイムスタンプを取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(timestamp)

# %% yamlを読み込み
with open("../hub/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# %%  データセットの読み込み
df_events = pd.read_csv(cfg["rawdata"]["event_path"])
df_ads = pd.read_csv(cfg["rawdata"]["ads_path"])
df_cps = pd.read_csv(cfg["rawdata"]["cps_path"])
df_users = pd.read_csv(cfg["rawdata"]["users_path"])

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

# 処理前のimp、click数を確認
print(f"raw imp:{df_merged["event_type_Impression"].sum()}")
print(f"raw click:{df_merged["event_type_Click"].sum()}")


########################################################
# 1. 前処理（分割前）
########################################################
# ============================================
# 1-1. 目的変数の処理
# ============================================
# %% imp列を追加（すべての値を1にする）
df_merged["imp"] = 1

# %% click列を追加（Purchase or event_type_clickが1の時、1を入れる。その他は0）
df_merged["click"] = np.where((df_merged["event_type_Click"] ==1) |(df_merged["event_type_Purchase"] ==1), 1, 0)

# %% Purchaseの列名を変更
df_merged = df_merged.rename(columns={"event_type_Purchase":"Purchase"})

# %% Engagement列を追加
df_merged["engagement"] = np.where((df_merged["event_type_Comment"]==1) |(df_merged["event_type_Like"]==1) |(df_merged["event_type_Share"]==1), 1, 0)

# %% CSVでdfを確認
# df_merged.head(100).to_csv(f"../outputs/df_merged_pre_event_chk_head100_{timestamp}.csv")


# ============================================
# 1-2. 全体の数値確認
# ============================================

# %% 全体のCTR・CVR・予算・CPC・CPAを算出

all_imps = df_merged["imp"].sum()
all_clicks = df_merged["click"].sum()
all_cvs = df_merged["Purchase"].sum()
all_costs = df_cps["total_budget"].sum()

ctr = all_clicks / all_imps
cvr = all_cvs / all_clicks
cpm = all_costs / all_imps *1000
cpc = all_costs / all_clicks
cpa = all_costs / all_cvs

print(f"imps:{all_imps},clicks:{all_clicks},cvs:{all_cvs},costs:{all_costs}\n")
print(f"ctr:{ctr},cvr:{cvr},cpm:{cpm},cpc:{cpc},cpa:{cpa}\n")



# ============================================
# 1-3. 特徴量の処理
# ============================================
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


# %% --- interestを変換
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

# %% CSVに前処理後のdfを格納
df_merged.to_csv(f"../outputs/df_merged_pretreatment_{timestamp}.csv")

# ============================================
# 1-4. 学習データとテストデータの分割
# ============================================
# %% 目的変数を設定せずに、Clickの比率を維持したまま分割
train_idx, test_idx = train_test_split(df_merged.index,test_size=0.2, random_state=0, stratify=df_merged["click"])

train_all = df_merged.loc[train_idx].reset_index(drop=True)
test_all = df_merged.loc[test_idx].reset_index(drop=True)


# %% 分割結果・確認データをCSVに保存
train_all.to_csv(f"../outputs/df_train_all_{timestamp}.csv")
test_all.to_csv(f"../outputs/df_test_all_{timestamp}.csv")

train_all.describe().to_csv(f"../outputs/chk_num_train_all_{timestamp}.csv")
train_all.describe(exclude='number').to_csv(f"../outputs/chk_cat_train_all_{timestamp}.csv")

test_all.describe().to_csv(f"../outputs/chk_num_test_all_{timestamp}.csv")
test_all.describe(exclude='number').to_csv(f"../outputs/chk_cat_test_all_{timestamp}.csv")
