import pandas as pd
import numpy as np

train_df = pd.read_csv("used_car_train_20200313.csv",index_col='SaleID',sep=' ')
test_df = pd.read_csv("used_car_testB_20200421.csv",index_col="SaleID", sep=' ')

data = pd.concat([train_df, test_df])

data1 = data.copy()

data1 = data1[data1['model'].notnull()]

for col in ['bodyType', 'fuelType', 'gearbox']:
    groupby_brand = data1.groupby(by='brand')[col].\
        agg(lambda x: x.value_counts().sort_values(ascending=False).index[0])
    nan_index = data1[data1[col].isnull()].index
    data1.loc[nan_index, col] = data1.loc[nan_index, 'brand'].map(lambda x: groupby_brand[x])

for col in ['model', 'bodyType', 'fuelType', 'gearbox']:
    data1[col] = data1[col].astype(int)

data1['notRepairedDamage'].replace('-', 2, inplace=True)
data1['notRepairedDamage'] = data1['notRepairedDamage'].astype(float).astype(int)
data1['region'] = data1['regionCode'].map(lambda x: str(x)[:2]).astype(int)
data1.drop([ 'regionCode'], axis=1, inplace=True)


data1.drop('seller', axis=1, inplace=True)
data1.drop("offerType", axis=1, inplace=True)


data1['reg_year'] = data1['regDate'].map(lambda x: x//10000)
data1['reg_month'] = data1['regDate'].map(lambda x: (x%10000)//100)
data1['reg_day'] = data1['regDate'].map(lambda x: x%100)
data1['reg_month'].replace(0,1, inplace=True)
data1['reg_day'].replace(0,1, inplace=True)

data1['create_year'] = data1['creatDate'].map(lambda x: x//10000)
data1['create_month'] = data1['creatDate'].map(lambda x: (x%10000)//100)
data1['create_day'] = data1['creatDate'].map(lambda x: x%100)
data1['create_month'].replace(0,1, inplace=True)
data1['create_day'].replace(0,1, inplace=True)

data1['regDate'] = pd.\
    to_datetime(data1['reg_year']*10000 + data1['reg_month']*100 + data1['reg_day'], format='%Y%m%d')
data1['creatDate'] = pd.\
    to_datetime(data1['create_year']*10000 + data1['create_month']*100 + data1['create_day'], format='%Y%m%d')

data1['used_time'] = (data1['creatDate'] - data1['regDate']).dt.days
data1.drop(['regDate', 'creatDate'], axis=1, inplace=True)


data1['power_'] = data1['power'].map(lambda x: x if x<=600 else 600)
data1.drop([ 'power'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

train = data1[data1['price'].notnull()]
x = train.drop('price', axis=1)
y = train['price']
test = data1[data1['price'].isnull()].drop('price', axis=1)
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2, random_state=6)

# 定义随机森林的参数网格
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# 使用网格搜索进行超参数调优
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
rf_grid_search.fit(train_x, np.log(train_y))
# 打印最佳参数组合和对应的得分
print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
print("Best Score for Random Forest:", -rf_grid_search.best_score_)  # 注意：网格搜索返回的是负的平均交叉验证分数

# 使用最佳参数组合训练随机森林模型
best_rf_params = rf_grid_search.best_params_
rfr_best = RandomForestRegressor(**best_rf_params)
rfr_best.fit(train_x, np.log(train_y))
y_hat = np.exp(rfr_best.predict(test_x))
mean_absolute_error(test_y, y_hat)
print('RandomForest score:{}'.format(mean_absolute_error(test_y, y_hat)))

# 定义XGBoost的参数网格
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.05, 0.1]
}

# 使用网格搜索进行超参数调优
xgb_grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), xgb_param_grid, cv=5)
xgb_grid_search.fit(train_x, np.log(train_y))

# 打印最佳参数组合和对应的得分
print("Best Parameters for XGBoost:", xgb_grid_search.best_params_)
print("Best Score for XGBoost:", -xgb_grid_search.best_score_)

# 使用最佳参数组合训练XGBoost模型
best_xgb_params = xgb_grid_search.best_params_
xgbr_best = xgb.XGBRegressor(objective='reg:squarederror', **best_xgb_params)
xgbr_best.fit(train_x, np.log(train_y))
# xgbr = xgb.XGBRegressor\
#     (n_estimators=500,max_depth=3,objective='reg:squarederror', n_jobs=2, learning_rate=0.35,reg_alpha=0.05)
# xgbr.fit(train_x, np.log(train_y))
y_hat_xgb = np.exp(xgbr_best.predict(test_x))
print('xgb score:{}'.format(mean_absolute_error(test_y, y_hat_xgb)))


class avg_model:
    def __init__(self, models):
        self.models = models

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)
        return self

    def predict(self, x):
        predictions = np.column_stack([ model.predict(x)
            for model in self.models ])
        return np.mean(predictions, axis=1)


model = avg_model([rfr_best, xgbr_best])
yhat = np.exp(model.predict(test_x))
print('mean model score:{}'.format(mean_absolute_error(test_y, yhat)))
ypred = np.exp(model.predict(test))

submission = pd.DataFrame(ypred, columns=['price'])
submission.index = test.index
submission.to_csv('bootpuss_submit.csv')
