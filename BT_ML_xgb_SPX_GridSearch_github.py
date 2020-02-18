'''

This uses GridSearchCV which has resulted in over-fitting

'''


import matplotlib
matplotlib.use('qt5agg') # Remove this to compare with MacOSX backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from ML_portfolio_rfr import run_split_train_test
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

from backtest import Portfolio,Strategy
from sklearn.metrics import accuracy_score



class xgb_predictive_SPX(Strategy):

    def __init__(self,stocks,df):
        self.stocks = stocks
        self.df = df

    def generate_signals(self):
        predvals_diff = self.run_xgb_model()
        sig = pd.DataFrame(index = predvals_diff.index)
        sig['signal'] = np.where(predvals_diff['pred']>0,1,-1)
        sig = pd.concat([sig,predvals_diff],axis=1)
        return sig

    def run_xgb_model(self):
        import xgboost as xgb
        from xgboost import XGBRegressor

        X = self.df.drop('spx', axis=1).iloc[:-1, :]
        y = self.df.spx.shift(-1).dropna()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=1,shuffle=False)

        #DM_train = xgb.DMatrix(data=X_train, label=y_train)
        #DM_test = xgb.DMatrix(data=X_test, label=y_test)

        xgbm = xgb.XGBRegressor()
        xgbm.fit(X_train,y_train)


        gbm_param_grid = {'learning_rate': [.01, .1, .5, .9],
                          'n_estimators': [200,300],
                          'subsample': [0.3, 0.5, 0.9],
                          'max_depth':[2,3],
                          'reg_lambda':[0]
                          }

        fit_params = {"early_stopping_rounds": 25,
                      "eval_metric": "rmse",
                      "eval_set": [(X_train, y_train), (X_test, y_test)]}

        #evals_result = {}
        #eval_s = [(X_train, y_train), (X_test, y_test)]


        tscv = TimeSeriesSplit(n_splits=2)
        xgb_Gridcv = GridSearchCV(estimator=xgbm, param_grid=gbm_param_grid, cv=tscv,refit = True,verbose=0)

        xgb_Gridcv.fit(X_train, y_train, **fit_params)
        ypred = xgb_Gridcv.predict(X_test)

        print(xgb_Gridcv.score(X_train, y_train))
        print(xgb_Gridcv.score(X_test, y_test))

        results = xgb_Gridcv.best_estimator_.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Regression Error')
        plt.show()


        print('best parameters',xgb_Gridcv.best_params_)
        print('Lowest RMSE', np.sqrt(np.abs(xgb_Gridcv.best_score_)))

        y_actual = pd.DataFrame(y_test)
        y_pred = pd.DataFrame(ypred)

        y_pred.index = y_actual.index
        pred = pd.concat([y_actual, y_pred], axis=1)
        pred.columns = ['actual', 'pred']
        #pred.sort_index(inplace=True)
        pred_vals = pred.shift(1)
        pred_vals_diff = pred_vals.pct_change()
        pred_vals_diff.index = pred_vals.index
        pred_vals_diff.columns = ['actual', 'pred']
        pred_vals_diff.dropna(inplace=True)
        return pred_vals_diff


class MarketonClosePortfolio(Portfolio):

    def __init__(self,sig):
        self.sig = sig
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = self.sig.drop(['actual','pred'],1)
        positions['retns'] = positions['signal'].mul(self.sig['actual'].values)
        return positions

    def backtest_portfolio(self):
        return(self.positions)




if __name__ == '__main__':

    spx = pd.read_csv('spx_long.csv', index_col='Date', parse_dates=True).dropna()['Close']
    crude = pd.read_csv('crude.csv',index_col='Date',parse_dates=True)
    dollar = pd.read_csv('dollar.csv',index_col='Date',parse_dates=True)['Settle']
    gold = pd.read_csv('gold.csv', index_col='Date', parse_dates=True)
    yc = pd.read_csv('yc.csv',index_col='Date',parse_dates=True)[['2 YR','10 YR']]
    hyOAS = pd.read_csv('usdhyOAS.csv',index_col='DATE',parse_dates=True)
    #senti = pd.read_csv('um_consumer.csv',index_col='Date',parse_dates=True)

    '''
    #rolling
    crude = crude.rolling(252).mean().dropna()
    dollar = dollar.rolling(252).mean().dropna()
    gold = gold.rolling(252).mean().dropna()
    yc = yc.rolling(252).mean().dropna()
    hyOAS = hyOAS.rolling(252).mean().dropna()
    '''

    #reindex to hyOAS
    spx = spx.reindex(hyOAS.index)
    crude = crude.reindex(hyOAS.index)
    dollar = dollar.reindex(hyOAS.index)
    gold = gold.reindex(hyOAS.index)
    yc = yc.reindex(hyOAS.index)

    #take term spread
    yc['10y2y'] = yc['10 YR'] - yc['2 YR']
    yc.drop(['2 YR', '10 YR'],axis=1,inplace=True)

    df = pd.DataFrame(index=spx.index)
    df = pd.concat([df,crude,dollar,gold,yc,spx],axis=1)

    stocknames = ['crude', 'dollar', 'gold', 'yc','spx']
    df.columns = stocknames
    df.ffill(inplace=True)

    xgb = xgb_predictive_SPX(stocknames,df)
    sig = xgb.generate_signals()

    mktport = MarketonClosePortfolio(sig)
    returns = mktport.backtest_portfolio()

    spx_1D = spx.pct_change()
    spx_1D = spx_1D.reindex(returns.index)


    #cumulative returns

    cum_xgb = np.cumproduct(1+returns.retns)-1
    cum_spx = np.cumproduct(1+spx_1D)-1

    combo = pd.DataFrame(index=cum_xgb.index)
    combo = pd.concat([cum_xgb,cum_spx],axis=1)
    combo.columns = ['xgb','spx']
    combo.plot()
    plt.legend(combo.columns)
    plt.title('L SPX vs XGB returns')
    plt.show()