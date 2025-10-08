import pandas as pd
import numpy as np


from sklearn.linear_model import  Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

class LassoModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rmse_train = None
        self.rmse_test = None

    def train_test(self):
        # filter invalid targets to avoid NaNs in log1p

        # if 'price_doc' in df_valid.columns:
        #     df_valid = df_valid[df_valid['price_doc'].notna() & (df_valid['price_doc'] > 0)]

        num_cols = self.df.select_dtypes(include='number').columns.tolist()

        X = self.df[num_cols].drop(columns='price_doc', axis=1)
        y = self.df['price_doc']

        # 80% train, 20% test with shuffle for better generalization
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42,
            shuffle=True,
        )

    def learn_model(self):
        print('Приступаю к обучению модели Lasso')


        params = {
            "Lasso__alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10]
        }

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('Lasso', Lasso()),

        ])

        model_search = GridSearchCV(pipe, params, verbose=0)
        model_search.fit(self.X_train, self.y_train)

        print(model_search.best_params_)

        train_pred = model_search.predict(self.X_train)
        test_pred = model_search.predict(self.X_test)

        self.rmse_train = np.sqrt(mean_squared_error(self.y_train, train_pred))
        self.rmse_test = np.sqrt(mean_squared_error(self.y_test, test_pred))


    def run(self):
        self.train_test()
        self.learn_model()


        return f'RMSE (train): {self.rmse_train} \nRMSE (test): {self.rmse_test} '


class ForestRandomModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rmse_train = None
        self.rmse_test = None


    def train_test(self):
        num_cols = self.df.select_dtypes(include='number').columns.tolist()

        X = self.df[num_cols].drop(columns='price_doc', axis=1)
        y = self.df['price_doc']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

    def learn_model(self):
        print('Приступаю к обучению модели RandomForest')

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('RF', RandomForestRegressor(
                n_estimators=1000,  # количество деревьев
                max_depth=None,  # глубина
                min_samples_split=4,  # минимум для разделения
                min_samples_leaf=2,  # минимум в листе
                max_features='sqrt',  # sqrt или log2
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipe.fit(self.X_train, self.y_train)



        train_pred = pipe.predict(self.X_train)
        test_pred = pipe.predict(self.X_test)
        self.rmse_train = np.sqrt(mean_squared_error(train_pred, self.y_train))
        self.rmse_test = np.sqrt(mean_squared_error(test_pred, self.y_test))

    def run(self):
        self.train_test()
        self.learn_model()

        return f'RMSE (train): {self.rmse_train} \nRMSE (test): {self.rmse_test}'


