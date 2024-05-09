import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

group = 9 #選擇組數

'''樣本內'''

perform = pd.DataFrame()
train_set = pd.DataFrame()
company_num = [0, ]
cost = 1.585
for i in range(1, 12): #11個季度為樣本內資料
  data = pd.read_csv("FS/"+str(i)+".csv", index_col='code') #引入財報特徵
  data = data.apply(pd.to_numeric, errors='coerce')
  data = data.drop(['name', 'return', 'annual_return'], axis=1)
  data = data.dropna()
  train_set = data
  #dummy化數據
  bins = [-np.inf, 0, np.inf]
  train_set['ROA_now'] = pd.cut(train_set['ROA_now'], bins, labels=[0, 1])
  train_set['CFO'] = pd.cut(train_set['CFO'], bins, labels=[0, 1])
  train_set['CFO-NI'] = pd.cut(train_set['CFO-NI'], bins, labels=[0, 1])
  train_set['NCL_chg'] = pd.cut(train_set['NCL_chg'], bins, labels=[1, 0])
  train_set['CR_chg'] = pd.cut(train_set['CR_chg'], bins, labels=[0, 1])
  train_set['CE_chg'] = pd.cut(train_set['CE_chg'], bins, labels=[1, 0])
  train_set['ROA_chg'] = pd.cut(train_set['ROA_chg'], bins, labels=[0, 1])
  train_set['MR_chg'] = pd.cut(train_set['MR_chg'], bins, labels=[0, 1])
  train_set['ATO_chg'] = pd.cut(train_set['ATO_chg'], bins, labels=[0, 1])
  train_set = train_set.apply(pd.to_numeric, errors='coerce')
  train_set = train_set.dropna()
  train_set['F'] = train_set['ROA_now']+train_set['CFO']+train_set['CFO-NI']+train_set['NCL_chg']+train_set['CR_chg']+train_set['CE_chg']+train_set['ROA_chg']+train_set['MR_chg']+train_set['ATO_chg']
  train_set = train_set.apply(pd.to_numeric, errors='coerce')
  train_set = train_set.dropna()
  train_set = train_set[(train_set['F']==group)]

  company = list(train_set.index)

  data_ret = pd.read_csv("Return/"+str(i+1)+".csv", index_col='t') #引入報酬率資料
  try:
    data_ret = data_ret.drop(None)
  except:
    pass
  data_ret = data_ret.apply(pd.to_numeric, errors='coerce')
  data_ret = data_ret.fillna(0)
  data_ret = data_ret.dropna()
  #交易成本
  data_ret.iloc[0] = data_ret.iloc[0] - cost
  print(company)
  data_p = pd.DataFrame()
  if company == []:
    data_p['return'] = data_p.apply(lambda x: 0, axis=1)
  else:
    for j in range(0, len(company)):
      company[j] = str(company[j])
      data_p[company[j]] = data_ret[company[j]]
      data_p[company[j]] = np.exp(data_ret[company[j]]/100)-1
    data_p['return'] = data_p.apply(lambda x: x.mean(), axis=1)
    data_p['return'] = np.log(1+data_p['return'])*100
  perform = pd.concat([perform, data_p['return']], axis=0, )
perform = perform.dropna()
perform_table = pd.DataFrame()
perform_table['ret'] = perform
perform_table['cum_ret'] = perform.cumsum()
perform_table['MDD'] = perform_table['cum_ret'].cummax() - perform_table['cum_ret']
#畫圖
fig, ax = plt.subplots(figsize=(16,6))
perform_table['cum_ret'].plot(label='Cumulative Return', ax=ax, c='b')
plt.fill_between(perform_table['MDD'].index, -perform_table['MDD'], 0, facecolor='r', label='DD')
plt.legend()
plt.ylabel('Return%')
plt.xlabel('Date')
plt.title('Return & MDD', fontsize=16)
plt.show()
#績效
annual_return = perform_table['ret'].mean() * 240
cum_return = perform_table['cum_ret'].iloc[-1]
annual_std = perform_table['ret'].std() * (240 ** 0.5)
MDD = round(perform_table['MDD'].max(),2)
print(f'年化報酬率: {round(annual_return, 2)}%') #年化報酬
print(f'累積報酬率: {round(cum_return, 2)}%') #累積報酬
print(f'年化波動度: {round(annual_std, 2)}%')
print(f'年化夏普值: {round(annual_return/annual_std, 2)}')
print(f'MDD(%): {MDD}%')
print(f'風報比: {round(cum_return / MDD, 2)}')

'''樣本外'''

perform = pd.DataFrame()
test_set = pd.DataFrame()
company_num = [0, ]
for i in range(12, 20): #8個季度為樣本外資料
  data = pd.read_csv("FS/"+str(i)+".csv", index_col='code') #引入財報特徵
  data = data.apply(pd.to_numeric, errors='coerce')
  data = data.drop(['name', 'return', 'annual_return'], axis=1)
  data = data.dropna()
  test_set = data
  #dummy化數據
  bins = [-np.inf, 0, np.inf]
  test_set['ROA_now'] = pd.cut(test_set['ROA_now'], bins, labels=[0, 1])
  test_set['CFO'] = pd.cut(test_set['CFO'], bins, labels=[0, 1])
  test_set['CFO-NI'] = pd.cut(test_set['CFO-NI'], bins, labels=[0, 1])
  test_set['NCL_chg'] = pd.cut(test_set['NCL_chg'], bins, labels=[1, 0])
  test_set['CR_chg'] = pd.cut(test_set['CR_chg'], bins, labels=[0, 1])
  test_set['CE_chg'] = pd.cut(test_set['CE_chg'], bins, labels=[1, 0])
  test_set['ROA_chg'] = pd.cut(test_set['ROA_chg'], bins, labels=[0, 1])
  test_set['MR_chg'] = pd.cut(test_set['MR_chg'], bins, labels=[0, 1])
  test_set['ATO_chg'] = pd.cut(test_set['ATO_chg'], bins, labels=[0, 1])
  test_set = test_set.apply(pd.to_numeric, errors='coerce')
  test_set = test_set.dropna()
  test_set['F'] = test_set['ROA_now']+test_set['CFO']+test_set['CFO-NI']+test_set['NCL_chg']+test_set['CR_chg']+test_set['CE_chg']+test_set['ROA_chg']+test_set['MR_chg']+test_set['ATO_chg']
  test_set = test_set.apply(pd.to_numeric, errors='coerce')
  test_set = test_set.dropna()
  test_set = test_set[(test_set['F']==group)]

  company = list(test_set.index)

  data_ret = pd.read_csv("Return/"+str(i+1)+".csv", index_col='t') #引入報酬率資料
  try:
    data_ret = data_ret.drop(None)
  except:
    pass
  data_ret = data_ret.apply(pd.to_numeric, errors='coerce')
  data_ret = data_ret.fillna(0)
  data_ret = data_ret.dropna()
  #交易成本
  data_ret.iloc[0] = data_ret.iloc[0] - cost
  print(company)
  data_p = pd.DataFrame()
  if company == []:
    data_p['return'] = data_p.apply(lambda x: 0, axis=1)
  else:
    for j in range(0, len(company)):
      company[j] = str(company[j])
      data_p[company[j]] = data_ret[company[j]]
      data_p[company[j]] = np.exp(data_ret[company[j]]/100)-1
    data_p['return'] = data_p.apply(lambda x: x.mean(), axis=1)
    data_p['return'] = np.log(1+data_p['return'])*100
  perform = pd.concat([perform, data_p['return']], axis=0, )
perform = perform.dropna()
perform_table = pd.DataFrame()
perform_table['ret'] = perform
perform_table['cum_ret'] = perform.cumsum()
perform_table['MDD'] = perform_table['cum_ret'].cummax() - perform_table['cum_ret']
#畫圖
fig, ax = plt.subplots(figsize=(16,6))
perform_table['cum_ret'].plot(label='Cumulative Return', ax=ax, c='b')
plt.fill_between(perform_table['MDD'].index, -perform_table['MDD'], 0, facecolor='r', label='DD')
plt.legend()
plt.ylabel('Return%')
plt.xlabel('Date')
plt.title('Return & MDD', fontsize=16)
plt.show()
#績效
annual_return = perform_table['ret'].mean() * 240
cum_return = perform_table['cum_ret'].iloc[-1]
annual_std = perform_table['ret'].std() * (240 ** 0.5)
MDD = round(perform_table['MDD'].max(),2)
print(f'年化報酬率: {round(annual_return, 2)}%') #年化報酬
print(f'累積報酬率: {round(cum_return, 2)}%') #累積報酬
print(f'年化波動度: {round(annual_std, 2)}%')
print(f'年化夏普值: {round(annual_return/annual_std, 2)}')
print(f'MDD(%): {MDD}%')
print(f'風報比: {round(cum_return / MDD, 2)}')