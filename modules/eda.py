########################################################
# 0. 事前準備（データの読み込みと確認）
########################################################

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
# 1. データ分布の可視化
########################################################

# %% --- 年齢分布確認
fig = sns.FacetGrid(train_all,col='event_type_Click',hue='event_type_Click',height=4)
fig.map(sns.histplot,'user_age',bins=100,kde=False)
plt.savefig(f'../outputs/push/figures/user_age_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- 性別分布確認
sns.countplot(x='user_gender',hue='event_type_Click',data=train_all)
plt.savefig(f'../outputs/push/figures/user_gender_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- 時間帯分布確認
fig = sns.FacetGrid(train_all,col='event_type_Click',hue='event_type_Click',height=4)
fig.map(sns.histplot,'hour',bins=100,kde=False)
plt.savefig(f'../outputs/push/figures/hour_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- 曜日分布確認
sns.countplot(x='day_of_week',hue='event_type_Click',data=train_all)
plt.savefig(f'../outputs/push/figures/day_of_week_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- 広告プラットフォーム分布確認
sns.countplot(x='ad_platform',hue='event_type_Click',data=train_all)
plt.savefig(f'../outputs/push/figures/ad_platform_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# %% --- 広告タイプ分布確認
sns.countplot(x='ad_type',hue='event_type_Click',data=train_all)
plt.savefig(f'../outputs/push/figures/ad_type_map_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()