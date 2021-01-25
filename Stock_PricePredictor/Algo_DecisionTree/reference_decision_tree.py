# STEP1：取得能夠定義預測目標的資料
# get_spy_data.py [https://gist.github.com/forcefintech/f8aae8de89bdad2b8f196d757dd1c1aa#file-get_spy_data-py]
import yfinance as yf
stk = yf.Ticker('SPY')
# 取得 2000 年至今的資料
data = stk.history(start = '2000-01-01')
# 簡化資料，只取開、高、低、收以及成交量
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]



# STEP2：準備更多拿來預測目標的因子
# talib_get_x.py [https://gist.github.com/forcefintech/c5fc7549dc83eb3658657eeee04ebff1#file-talib_get_x-py]
import pandas as pd
from talib import abstract
# 改成 TA-Lib 可以辨識的欄位名稱
data.columns = ['open','high','low','close','volume']
# 隨意試試看這幾個因子好了
ta_list = ['MACD','RSI','MOM','STOCH']
# 快速計算與整理因子
for x in ta_list:
    output = eval('abstract.'+x+'(data)')
    output.name = x.lower() if type(output) == pd.core.series.Series else None
    data = pd.merge(data, pd.DataFrame(output), left_on = data.index, right_on = output.index)
    data = data.set_index('key_0')



# STEP3：標記預測目標
# label_y.py [https://gist.github.com/forcefintech/7b2482de4eaf23e2ac9d61eea342a388#file-label_y-py]
import numpy as np
# 五日後漲標記 1，反之標記 0
data['week_trend'] = np.where(data.close.shift(-5) > data.close, 1, 0)

# plot_y.py [https://gist.github.com/forcefintech/b1850b5cb2bc0bef1307c766f58987fa#file-plot_y-py]
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
df = data['2019'].copy()
df = df.resample('D').ffill()
t = mdates.drange(df.index[0], df.index[-1], dt.timedelta(hours = 24))
y = np.array(df.close[:-1])
fig, ax = plt.subplots()
ax.plot_date(t, y, 'b-', color = 'black')
for i in range(len(df)):
    if df.week_trend[i] == 1:
        ax.axvspan(
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) - 0.5,
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) + 0.5,
            facecolor = 'green', edgecolor = 'none', alpha = 0.5
            )
    else:
        ax.axvspan(
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) - 0.5,
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) + 0.5,
            facecolor = 'red', edgecolor = 'none', alpha = 0.5
            )
fig.autofmt_xdate()
fig.set_size_inches(20, 10.5)
fig.savefig('define_y.png')



# STEP4：資料預處理
# nan_drop.py [https://gist.github.com/forcefintech/b1850b5cb2bc0bef1307c766f58987fa#file-plot_y-py]
# 檢查資料有無缺值
data.isnull().sum()
# 最簡單的作法是把有缺值的資料整列拿掉
data = data.dropna()



# STEP5：學習 / 測試樣本切割
# ml_split.py [https://gist.github.com/forcefintech/4d12dd0acc1527d64e4814809d237909#file-ml_split-py]
# 決定切割比例為 70%:30%
split_point = int(len(data)*0.7)
# 切割成學習樣本以及測試樣本
train = data.iloc[:split_point,:].copy()
test = data.iloc[split_point:-5,:].copy()

# X_y_split.py [https://gist.github.com/forcefintech/2a46581dcbb96c127d713e2747f5ace5#file-x_y_split-py]
# 訓練樣本再分成目標序列 y 以及因子矩陣 X
train_X = train.drop('week_trend', axis = 1)
train_y = train.week_trend
# 測試樣本再分成目標序列 y 以及因子矩陣 X
test_X = test.drop('week_trend', axis = 1)
test_y = test.week_trend



# STEP6：匯入 sklearn，創造一個決策樹
# decision_tree_import.py [https://gist.github.com/forcefintech/099e151d677164f080d3e3b0ccb1182b#file-decision_tree_import-py]
# 匯入決策樹分類器
from sklearn.tree import DecisionTreeClassifier
# 叫出一棵決策樹
model = DecisionTreeClassifier(max_depth = 7)



# STEP7：讓決策樹學習與測驗
# decision_tree_train_test.py [https://gist.github.com/forcefintech/8e17f29f28eaa98ef183b52b291d5fff#file-decision_tree_train_test-py]
# 讓 A.I. 學習
model.fit(train_X, train_y)
# 讓 A.I. 測驗，prediction 存放了 A.I. 根據測試集做出的預測
prediction = model.predict(test_X)

# visualize_decision_tree.py [https://gist.github.com/forcefintech/56ffcaf74c64b9c45389edee9c1c1a50#file-visualize_decision_tree-py]
# below method seems only applicable to Jupyter Environment
# from sklearn.tree import export_graphviz
# import graphviz
# dot_data = export_graphviz(model, out_file = None,
#                           feature_names = train_X.columns,
#                           filled = True, rounded = True,
#                           class_names = True,
#                           special_characters = True)
# graph = graphviz.Source(dot_data)
# graph

# new method from online to view the svm tree
from sklearn.tree import export_graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydotplus
dot_data = export_graphviz(model, out_file = None,
                           feature_names = train_X.columns,
                           filled = True, rounded = True,
                           class_names = True,
                           special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('decision_tree.pdf')



# STEP8：看看混淆矩陣，計算準確率
# confusion_matrix.py [https://gist.github.com/forcefintech/80941b9d34802680aa39b574d26a0917#file-confusion_matrix-py]
# 要計算混淆矩陣的話，要從 metrics 裡匯入 confusion_matrix
from sklearn.metrics import confusion_matrix
# 混淆矩陣
confusion_matrix(test_y, prediction)
# 準確率
model.score(test_X, test_y)



# STEP9：計算 AUC
# roc_auc.py [https://gist.github.com/forcefintech/721f8cd6fce4d85538e6b109758defb8#file-roc_auc-py]
# 要計算 AUC 的話，要從 metrics 裡匯入 roc_curve 以及 auc
from sklearn.metrics import roc_curve, auc
# 計算 ROC 曲線
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, prediction)
# 計算 AUC 面積
auc(false_positive_rate, true_positive_rate)


# STEP10：透過 AUC，決定決策樹深度的最佳參數
# find_depth_para.py [https://gist.github.com/forcefintech/08ba4f9b0a7bc8338130d7e21c9a5a56#file-find_depth_para-py]
import matplotlib.pyplot as plt
# 測試一批深度參數，一般而言深度不太會超過 3x，我們這邊示範 1 到 50 好了
depth_parameters = np.arange(1, 50)
# 準備兩個容器，一個裝所有參數下的訓練階段 AUC；另一個裝所有參數下的測試階段 AUC
train_auc= []
test_auc = []
# 根據每一個參數跑迴圈
for test_depth in depth_parameters:
    # 根據該深度參數，創立一個決策樹模型，取名 temp_model
    temp_model = DecisionTreeClassifier(max_depth = test_depth)
    # 讓 temp_model 根據 train 學習樣本進行學習
    temp_model.fit(train_X, train_y)
    # 讓學習後的 temp_model 分別根據 train 學習樣本以及 test 測試樣本進行測驗
    train_predictions = temp_model.predict(train_X)
    test_predictions = temp_model.predict(test_X)
    # 計算學習樣本的 AUC，並且紀錄起來
    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_predictions)
    auc_area = auc(false_positive_rate, true_positive_rate)
    train_auc.append(auc_area)
    # 計算測試樣本的 AUC，並且紀錄起來
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, test_predictions)
    auc_area = auc(false_positive_rate, true_positive_rate)
    test_auc.append(auc_area)

# 繪圖視覺化
plt.figure(figsize = (14,10))
plt.plot(depth_parameters, train_auc, 'b', label = 'Train AUC')
plt.plot(depth_parameters, test_auc, 'r', label = 'Test AUC')
plt.ylabel('AUC')
plt.xlabel('depth parameter')
plt.show()


# STEP11：簡易回測與績效計算
# strategy_test.py [https://gist.github.com/forcefintech/ac351efc13c83e057f195acb369a2243#file-strategy_test-py]
# test 是我們在切割樣本的時候，切出來的測試樣本，包含了價量資訊，我們首先將 A.I. 在這期間的預測結果 prediction 放進去
test['prediction'] = prediction

# 這次的二元分類問題很單純，若直接把 prediction 位移一天，剛好就會是模擬買賣的狀況：
# T-1 日的預測為「跌」而 T 日的預測為「漲」，則 T+1 日開盤『買進』
# T-1 日的預測為「漲」而 T 日的預測為「跌」，則 T+1 日開盤『賣出』
# 連續預測「漲」，則『持續持有』
# 連續預測「跌」，則『空手等待』
test['status'] = test.prediction.shift(1).fillna(0)

# 所以什麼時候要買股票就很好找了：status 從 0 變成 1 的時候，1 的那天的開盤買進（因為 status 已經位移一天了喔）
# 從 prediction 的角度解釋就是：當 A.I. 的預測從 0 變成 1 的時候，1 的隔天的開盤買進
test['buy_cost'] = test.open[np.where((test.status == 1) * (test.status.shift(1) == 0))[0]]
# 同理，賣股票也很好找：status 從 1 變成 0 的時候，0 的那天的開盤賣出
test['sell_cost'] = test.open[np.where((test.status == 0) * (test.status.shift(1) == 1))[0]]
# 把缺值補上 0
test = test.fillna(0)

# 來算算每次買賣的報酬率吧！
# 一買一賣是剛好對應的，所以把買的成本以及賣的價格這兩欄的數字取出，就能輕易的算出交易報酬率
# 但是回測的最後一天，會發現還有持股尚未賣出喔！由於還沒賣就不能當作一次完整的交易，所以最後一次的買進，我們先忽略
buy_cost = np.array(test.buy_cost[test.buy_cost != 0])[:-1]
sell_price = np.array(test.sell_cost[test.sell_cost != 0])
trade_return = sell_price / buy_cost - 1

# 交易都會有交易成本，例如台股每次一買一賣約產生 0.6% 的交易成本。
# 買賣 SPY ETF 也會有交易成本，管理費用約 0.1%，券商手續費因人而異，但近年來此費用逐漸趨近於 0，這裡就假設 0.1% 手續費好了
# 因此這裡額外計算一個把每次交易報酬率扣除總交易成本約 0.2% 的淨報酬率
fee = 0.002
net_trade_return = trade_return - fee

# 把報酬率都放進表格吧！
test['trade_ret'] = 0
test['net_trade_ret'] = 0
sell_dates = test.sell_cost[test.sell_cost != 0].index
test.loc[sell_dates, 'trade_ret'] = trade_return
test.loc[sell_dates, 'net_trade_ret'] = net_trade_return

# 如果還想要畫出績效走勢圖，那就要把策略的報酬率也算出來，由於我們不論買賣都是以開盤價進行，所以策略的報酬率會使用開盤價計算
test['open_ret'] = test.open / test.open.shift(1) - 1
test['strategy_ret'] = test.status.shift(1) * test.open_ret
test['strategy_net_ret'] = test.strategy_ret
test.loc[sell_dates, 'strategy_net_ret'] = test.loc[sell_dates, 'strategy_net_ret'] - fee
test = test.fillna(0)

# 計算出績效走勢圖
test['buy_and_hold_equity'] = (test.open_ret + 1).cumprod()
test['strategy_equity'] = (test.strategy_ret + 1).cumprod()
test['strategy_net_equity'] = (test.strategy_net_ret + 1).cumprod()

# 計算出一些有用的策略績效數字吧！
trade_count = len(sell_dates)
trade_count_per_year = trade_count / (len(test)/252)
win_rate = (net_trade_return > 0).sum() / trade_count
profit_factor = net_trade_return[net_trade_return > 0].sum() / abs(net_trade_return[net_trade_return < 0].sum())
mean_net_return = np.mean(net_trade_return)
acc_ret = test.strategy_net_equity[-1] - 1
strategy_ear = test.strategy_net_equity[-1] ** (252/len(test)) - 1
strategy_std = test.strategy_net_ret.std() * (252 ** 0.5)
strategy_sharpe = (strategy_ear - 0.01) / strategy_std

# 也畫出績效走勢看看吧！
test.buy_and_hold_equity.plot()
test.strategy_equity.plot()
test.strategy_net_equity.plot()


# STEP12：存 / 取 A.I. 模型
# pickle.py [https://gist.github.com/forcefintech/279c28876d8859fb7a89b41af859a23f#file-pickle-py]
import pickle

# 儲存模型
filename = 'Decision_Tree_model_001.sav'
pickle.dump(model, open(filename, 'wb'))

# 取出模型
filename = 'Decision_Tree_model_001.sav'
model = pickle.load(open(filename, 'rb'))