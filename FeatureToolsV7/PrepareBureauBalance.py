# coding:utf-8

import os
import pandas as pd
import numpy as np
import competition_utils as cu


class PrepareBureauBalance(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__bureau_balance = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

        # One-hot encoding for categorical columns with get_dummies
    def one_hot_encoder(df, nan_as_category = True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    def data_prepare(self):
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"))

    def data_transform(self):
        self.__bureau_balance = cu.replace_day_outliers(self.__bureau_balance)
        #self.__bureau_balance = self.__bureau_balance.replace(['XNA', 'XAP'], np.nan)

        self.__bureau_balance["TIME_MONTHS_BALANCE"] = pd.to_timedelta(self.__bureau_balance["MONTHS_BALANCE"], "M")

        self.__bureau_balance["TIME_MONTHS_BALANCE"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__bureau_balance.columns.tolist():
            if col in self.__bureau_balance.select_dtypes(include="object").columns.tolist():
                self.__bureau_balance.rename(columns={col: "FLAG_BUREAU_BALANCE_" + col}, inplace=True)

        self.__bureau_balance = pd.get_dummies(
            data=self.__bureau_balance,
            prefix="FLAG_BUREAU_BALANCE",
            dummy_na=True,
            columns=self.__bureau_balance.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        pass

    def data_return(self):
        # print(self.__bureau_balance.shape)
        self.__bureau_balance.to_csv(os.path.join(self.__input_path, "bureau_balance_temp.csv"), index=False)

        return self.__bureau_balance


if __name__ == "__main__":
    pbb = PrepareBureauBalance(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    pbb.data_prepare()
    pbb.data_transform()
    pbb.data_generate()
    pbb.data_return()
