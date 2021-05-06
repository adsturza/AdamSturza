import os

import pyforest
import pandas as pd
import datetime as dt
from lazypredict.Supervised import LazyRegressor
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings(action = 'ignore')

pd.options.display.max_columns = 100

root_path = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(root_path, 'kc_house_data.csv')

df = pd.read_csv(data, index_col=0)

#print(df.head)
#print(df.describe())
#print(df.info())

#cleanup
#df['date'] = pd.to_datetime(df['date'], format="%Y%m%dT%H%M%S")
df.drop('date', inplace=True, axis=1)
#print(df.head)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['bathrooms'] = df.bedrooms.apply(lambda x: 1 if x < 1 else x)

#splitting up train/test
X = df.drop(columns=['price'])
y = df.price
# Call train_test_split on the data and capture the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

def rmse(model, y_test, y_pred, X_train, y_train):
    r_squared = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('R-squared: ' + str(r_squared))
    print('Mean Squared Error: ' + str(rmse))


def scatter_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(10,6))
    sns.residplot(y_test, y_pred, lowess=True, color='#4682b4',
              line_kws={'lw': 2, 'color': 'r'})
    plt.title(str('Price vs Residuals for '+ model_name))
    plt.xlabel('Price',fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()

hist = HistGradientBoostingRegressor()
hist.fit(X_train, y_train)
y_pred = hist.predict(X_test)

rmse(hist, y_test, y_pred, X_train, y_train)
scatter_plot(y_test, y_pred, 'Histogram-based Gradient Boosting Regression Tree')