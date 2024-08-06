import pandas as pd
from pipeline import *
from utils import *

def process_train_test(train,test):
    train = train.reset_index(drop=True)
    train = train.apply(pd.to_numeric, errors='coerce')

    test = test.reset_index(drop=True)
    test = test.apply(pd.to_numeric, errors='coerce')

    x_train = train.drop('Passengers', axis=1)
    y_train = train['Passengers']

    test = test.reset_index(drop=True)
    x_test = test.drop('Passengers', axis=1)
    y_test = test['Passengers']

    x_train['weekofyear'] = x_train['weekofyear'].astype(int)
    x_test['weekofyear'] = x_test['weekofyear'].astype(int)

    return x_train, y_train, x_test, y_test


def main():

    path = extract()

    df = (pd.DataFrame()
            .pipe(lambda df: load_data(df, path))
            .pipe(set_index)
            .pipe(lambda df: rename(df,columns={'#Passengers': 'Passengers'}))
            .pipe(viz_data))
    
    print(df.head())

    train,test = data_split(df)
    viz_train_test(train,test)

    x_train, y_train, x_test, y_test = process_train_test(train,test)

    model = create_model(estimators=1000,learning_rate=0.01,max_depth=3,stop_rounds=100)

    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=100)

    viz_feature_imp(model)

if __name__ == "__main__":
    main()