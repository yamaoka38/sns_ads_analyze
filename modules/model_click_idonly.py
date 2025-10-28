########################################################
# 0. 事前準備（データの読み込みと確認）
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
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, roc_auc_score, log_loss
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
# %% 出力フォルダを指定
output_dir = Path("../outputs")

# ファイル名のパターン（例：user_clusters_20251024_1530.csv）
pattern = re.compile(r"df_train_all_user-ad_cluster_id_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files = sorted([f for f in output_dir.glob("df_train_all_user-ad_cluster_id_*.csv") if pattern.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files:
    raise FileNotFoundError("df_train_all_user-ad_cluster_id_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

# 最新ファイルを選択（ファイル名昇順の最後＝最新）
latest_file = files[-1]
print(f"読み込み対象ファイル: {latest_file.name}")

# CSVを読み込み
train_all = pd.read_csv(latest_file)    

# ============================================
# 0-3. テストデータを読み込み
# ============================================
# %% ファイル名のパターン（例：user_clusters_20251024_1530.csv）
pattern_test = re.compile(r"df_test_all_add_ctrcvr_\d{8}_\d{4}\.csv$")

# フォルダ内のファイルを検索 → パターンに一致するものだけ取得
files_test = sorted([f for f in output_dir.glob("df_test_all_add_ctrcvr_*.csv") if pattern_test.search(f.name)])

# ファイルが見つからなかった場合はエラー
if not files_test:
    raise FileNotFoundError("df_test_all_add_ctrcvr_YYYYMMDD_HHMM.csv の形式に合うファイルが見つかりません。")

# 最新ファイルを選択（ファイル名昇順の最後＝最新）
latest_file_test = files_test[-1]
print(f"読み込み対象ファイル(テスト): {latest_file_test.name}")

# CSVを読み込み
test_all = pd.read_csv(latest_file_test)

# 読み込みテスト
print(train_all.head(5))
print(test_all.head(5))

########################################################
# 1. 前処理
########################################################

train_all.columns.to_series().to_csv(f"../outputs/train_all_columnslist.csv")

# ============================================
# 1-1. 訓練データに広告IDの平均CTR・CVRを追加
# ============================================
ad_stats = (
    train_all.groupby("ad_id")
    .agg(
        avg_ctr=("click", lambda x: x.sum() / train_all.loc[x.index, "imp"].sum()),
        avg_cvr=("Purchase", lambda x: x.sum() / train_all.loc[x.index, "click"].sum())
    )
    .reset_index()
)

train_all = train_all.merge(ad_stats, on="ad_id", how="left")
print(train_all.head(10))


# ============================================
# 1-2. 使用するカラムを選定（まずは簡易走行）
# ============================================

use_cols=[
"click",
"user_cluster_id",
"ad_cluster_id"
]

pre_train_s = train_all[use_cols]
pre_train = pre_train_s.copy()

'''
# ============================================
# 1-3. 学習データの前処理
# ============================================

# %% --- 曜日を周期エンコーディング
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
pre_train["weekday_num"] = pre_train["day_of_week"].map(weekday_map)

pre_train["weekday_sin"] = np.sin(2 * np.pi * pre_train["weekday_num"] / 7)
pre_train["weekday_cos"] = np.cos(2 * np.pi * pre_train["weekday_num"] / 7)
pre_train = pre_train.drop(columns=["weekday_num","day_of_week"])
#pre_train.head(10).to_csv(f"../outputs/pre_train_weekday_fqenc_{timestamp}.csv")

# %% --- target_interestをワンホットエンコーディング
## カンマ区切りをリストに変換
pre_train["t_interests_list"] = pre_train["target_interests"].str.split(",")
## リストをワンホットエンコーディング
df_t_interests = pre_train["t_interests_list"].explode().str.strip().str.get_dummies().rename(columns=lambda col: f"t_{col}").groupby(level=0).sum()
print("df_t_interests")
print(df_t_interests.head(10))
## 元のdfに結合
pre_train = pd.concat([pre_train, df_t_interests], axis=1)
pre_train = pre_train.drop(["t_interests_list","target_interests"],axis=1)
#pre_train.head(10).to_csv(f"../outputs/pre_train_tint_enc_{timestamp}.csv")


# %% --- カテゴリカル変数をワンホットエンコーディング
cat_cols = ["ad_platform","ad_type","target_gender","user_gender"]
pre_train = pd.get_dummies(pre_train, columns=cat_cols,drop_first=False,dtype=int)
# pre_train.to_csv(f"../outputs/pre_train_encoded_{timestamp}.csv")
#print(f"ワンホットエンコーディング後：{pre_train.describe()}")
#pre_train.head(10).to_csv(f"../outputs/pre_train_catcols_enc_{timestamp}.csv")
'''

# %% --- 数値変数を標準化
scaler = StandardScaler()
num_cols = ["user_cluster_id","ad_cluster_id"]
pre_train[num_cols] = scaler.fit_transform(pre_train[num_cols])
pre_train.head(10).to_csv(f"../outputs/pre_train_scalered_{timestamp}.csv")
print(f"標準化後：{pre_train.describe()}")

# ============================================
# 1-4. 学習データ分割
# ============================================

# --- 目的変数を定義
x = pre_train.drop(columns = ["click"])
y = pre_train["click"]

# --- データの20%を検証データに変更(clickの偏りを均等に)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0, stratify=y
)

# --- 残りのデータをクロスバリデーション
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # クラス比率を維持して分割

########################################################
# 2. クリック予測のモデル構築と評価（複数モデル評価）
########################################################

# ============================================
# 2-1. モデル定義
# ============================================
models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=0,
        n_jobs=-1
    ),
}

########################################################
# 3. モデル精度の確認
########################################################

scoring = {"AUC":"roc_auc", "Logloss":"neg_log_loss" }

# 評価ループ
rows = []
for name, est in models.items():
    scores = cross_validate(
        est, X_train, y_train,
        cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=True
    )
    rows.append({
        "model": name,
        # AUC（大きいほど良い）
        "train_AUC_mean": np.mean(scores["train_AUC"]),
        "test_AUC_mean":  np.mean(scores["test_AUC"]),
        # Logloss（小さいほど良い）→ 負で返るので符号を戻す
        "train_Logloss_mean": -np.mean(scores["train_Logloss"]),
        "test_Logloss_mean":  -np.mean(scores["test_Logloss"]),
         # 参考: 計算時間
        "fit_time_mean":  np.mean(scores["fit_time"]),
        "score_time_mean":np.mean(scores["score_time"]),
    })

df_cv = pd.DataFrame(rows).sort_values("test_AUC_mean", ascending=False).reset_index(drop=True)
pd.set_option("display.max_columns", None)
print(df_cv.round(4))
