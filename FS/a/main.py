import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance
from sklearn.compose import ColumnTransformer
import shap

train_set = pd.DataFrame()
for i in range(1, 16): #15個季度為樣本內資料
  data = pd.read_csv(""+str(i)+".csv", index_col='code')
  data = data.replace('#DIV/0!', None)
  data = data.drop(['name', ], axis=1)
  data = data.dropna()
  train_y = data['return']
  data = data.drop(['return', ], axis=1)
  # 保存特徵名稱，因為標準化後將會失去
  feature_names = data.columns
  #標準化數據
  scaler = StandardScaler()
  data = scaler.fit_transform(data)
  # 將標準化後的訓練數據轉換回DataFrame格式，並加入列名
  data = pd.DataFrame(data, columns=feature_names)
  data = pd.concat([data, train_y], axis=1)
  train_set = pd.concat([train_set, data], axis=0, )
  train_set = train_set.dropna()
  #train_y = train_set['return']
  train_set = train_set.drop(['return', ], axis=1)

test_set = pd.DataFrame()
for i in range(16, 20): #4個季度為樣本外資料
  data = pd.read_csv(""+str(i)+".csv", index_col='code')
  data = data.replace('#DIV/0!', None)
  data = data.drop(['name', ], axis=1)
  data = data.dropna()
  train_y = data['return']
  data = data.drop(['return', ], axis=1)
  # 保存特徵名稱，因為標準化後將會失去
  feature_names = data.columns
  #標準化數據
  scaler = StandardScaler()
  data = scaler.fit_transform(data)
  # 將標準化後的訓練數據轉換回DataFrame格式，並加入列名
  data = pd.DataFrame(data, columns=feature_names)
  data = pd.concat([data, train_y], axis=1)
  test_set = pd.concat([test_set, data], axis=0, )
  test_set = test_set.dropna()
  #test_y = test_set['return']
  test_set = test_set.drop(['return', ], axis=1)

#分類模型（監督式）
models = {
    #"Decision Tree" : DecisionTreeRegressor(),
    "SVM" : LinearSVR(epsilon=0.1),
    #Bagging
    #"RandomForest" : RandomForestRegressor(),
    #Boosting
    #"XGBoost" : xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=1, gamma=0.1),
}

score_list_train = []
score_list_test = []
i = 0
for model_name, model in models.items():
  #訓練模型
  model.fit(train_set, train_y)

  #預測訓練集
  pred_y = model.predict(train_set)

  #預測測試集
  pred_y_test = model.predict(test_set)

  #計算評分
  score_list_train.insert(i, [mean_absolute_error(train_y, pred_y), mean_squared_error(train_y, pred_y), mean_squared_error(train_y, pred_y, squared=False)])
  print(f"{model_name}:\n {score_list_train[i]}")

  score_list_test.insert(i, [mean_absolute_error(test_y, pred_y_test), mean_squared_error(test_y, pred_y_test), mean_squared_error(test_y, pred_y_test, squared=False)])
  print(f"{model_name}:\n {score_list_test[i]}")

  try:
    explainer = shap.Explainer(model)
    shap_values = explainer(train_set)
    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values)
    # Bar Chart
    shap.plots.bar(shap_values)
  except:
    print('不適用')

  i += 1

#測試集分布


max = pred_y_test.max()
min = pred_y_test.min()
bin = (max - min) / 10
bins = [min, min+bin, min+bin*2, min+bin*3, min+bin*4, min+bin*5, min+bin*6, min+bin*7, min+bin*8, min+bin*9, max]

pred_y_test = pd.Series(pred_y_test)
pred_y_test = pd.cut(pred_y_test, bins, labels=[0,1,2,3,4,5,6,7,8,9])
test_table = pd.DataFrame({'group':pred_y_test, 'return':test_y})
test_table = test_table.dropna()
test_table = test_table.groupby('group').sum()
print(test_table)