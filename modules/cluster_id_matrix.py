########################################################
# 0. 事前準備（データの読み込み）
########################################################
# ============================================
# 0-1. 必要なモジュールのインポート
# ============================================
# %% 必要なモジュールのインポート
import yaml
from datetime import datetime
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%  タイムスタンプを取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(timestamp)

# %% yamlを読み込み
with open("../hub/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ============================================
# 0-2. データを読み込み
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

########################################################
# 1. 集計処理
########################################################

# クラスタIDの掛け合わせ毎に成果指標を集計
agg = (
    train_all.groupby(["user_cluster_id","ad_cluster_id"],as_index=False)
    .agg(
        imp=("imp","sum"),ct=("click","sum"),cv=("Purchase","sum")
    )
)

# CTR・CVRを計算
agg["ctr"] = agg["ct"] / agg["imp"].replace({0:np.nan})
agg["cvr"] = agg["cv"] / agg["ct"].replace({0:np.nan})
agg["ctvr"] = agg["ctr"] * agg["cvr"].replace({0:np.nan})


#print(agg)

# ピボット化

pivot_ctr = agg.pivot(index="user_cluster_id", columns="ad_cluster_id", values="ctr")
pivot_cvr = agg.pivot(index="user_cluster_id", columns="ad_cluster_id", values="cvr")
pivot_ctvr = agg.pivot(index="user_cluster_id", columns="ad_cluster_id", values="ctvr")
pivot_clicks = agg.pivot(index="user_cluster_id", columns="ad_cluster_id", values="ct")
pivot_cv = agg.pivot(index="user_cluster_id", columns="ad_cluster_id", values="cv")

pivot_ctr.to_csv(f"../outputs/push/claster_matrix_ctr_{timestamp}.csv")
pivot_cvr.to_csv(f"../outputs/push/claster_matrix_cvr_{timestamp}.csv")
pivot_ctvr.to_csv(f"../outputs/push/claster_matrix_ctvr_{timestamp}.csv")


# -- ヒートマップ画像化
fig, axes = plt.subplots(3,1,figsize=(5,10))

sns.heatmap(
    pivot_ctr,
    cmap="coolwarm",
    annot=True,
    fmt=".3f",
    linewidth=0.5,
    cbar=True,
    ax=axes[0]
)
axes[0].set_title("cluster_matrix_ctr",fontsize=14)

sns.heatmap(
    pivot_cvr,
    cmap="coolwarm",
    annot=True,
    fmt=".3f",
    linewidth=0.5,
    cbar=True,
    ax=axes[1]
)
axes[1].set_title("cluster_matrix_cvr",fontsize=14)

sns.heatmap(
    pivot_ctvr,
    cmap="coolwarm",
    annot=True,
    fmt=".4f",
    linewidth=0.5,
    cbar=True,
    ax=axes[2]
)
axes[2].set_title("cluster_matrix_ctvr",fontsize=14)

plt.tight_layout()

# ④ 画像として保存
plt.savefig(f"../outputs/push/figures/cluster_matrix_heatmap_{timestamp}.png", dpi=150)
plt.show()