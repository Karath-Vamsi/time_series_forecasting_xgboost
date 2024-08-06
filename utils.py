import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

def data_split(df):

    train_size = int(len(df) * 0.75)
    train, test = df[0:train_size], df[train_size:]
    return train, test

def viz_train_test(train,test):

    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
    test.plot(ax=ax, label='Test Set')
    ax.legend(['Training Set', 'Test Set'])
    plt.show()

def create_features(df):

    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    features = df[:-1].columns
    target = df[-1]

    return df,features,target

def create_model(estimators,learning_rate,max_depth,stop_rounds): 

    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=estimators,
                       early_stopping_rounds=stop_rounds,
                       objective='reg:linear',
                       max_depth=max_depth,
                       learning_rate=learning_rate)
    
    return model

def viz_feature_imp(model):

    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='weight', title='Feature Importance', xlabel='F Score', ylabel='Features')
    plt.show()
