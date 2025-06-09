import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

wine = load_wine() #Wineデータセットの読み込み
X, y = wine.data, wine.target
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names) #DataFrameとして変換する

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

feat_labels = df_wine.columns[:] #Wineデータセットの特徴量の名称

tree_counts=[10, 50, 500, 1000] #課題条件

for i in tree_counts:
    forest = RandomForestClassifier(n_estimators=i, random_state=1) #ランダムフォレストオブジェクトの作成(決定木の数=500,シード1)
    forest.fit(X_train, y_train) #モデルを適合(学習)
    importances = forest.feature_importances_ #特徴量の重要度を抽出
    indices = np.argsort(importances)[::-1] #重要度の降順で特徴量の名称重要度を表示する
    
    for f in range(X_train.shape[1]): #重要度の降順で特徴量の名称、重要度を表示
        print("%2d) %-*s %f" % 
            (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
    plt.title(f"Feature importances tree_count={i}")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1,X_train.shape[1]])
    plt.tight_layout()
    plt.show()