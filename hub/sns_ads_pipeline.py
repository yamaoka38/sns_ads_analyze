import subprocess as sub

print("rawdataの読み込み・マージ・前処理を実行")
sub.run(["python","../modules/merge_and_pretreatment.py"], check=True)

print("ユーザー特徴量でのクラスタリングを実行")
sub.run(["python","../modules/clustering_user.py"], check=True)

print("広告特徴量でのクラスタリングを実行")
sub.run(["python","../modules/clustering_ad.py"], check=True)

print("全特徴量でのclick予測を実行")
sub.run(["python","../modules/model_click.py"], check=True)

print("実行完了")