# coding:utf-8

import os
import pandas as pd
import numpy as np
import competition_utils as cu


class PreparePosCash(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__pos_cash = None

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
        self.__pos_cash = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        self.__pos_cash = self.__pos_cash.drop(["SK_ID_CURR"], axis=1)

    def data_transform(self):
        self.__pos_cash = cu.replace_day_outliers(self.__pos_cash)
        #self.__pos_cash = self.__pos_cash.replace(['XNA', 'XAP'], np.nan)

        self.__pos_cash["TIME_MONTHS_BALANCE"] = pd.to_timedelta(self.__pos_cash["MONTHS_BALANCE"], "M")

        self.__pos_cash["TIME_MONTHS_BALANCE"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__pos_cash.columns.tolist():
            if col in self.__pos_cash.select_dtypes(include="object").columns.tolist():
                self.__pos_cash.rename(columns={col: "FLAG_POS_CASH_" + col}, inplace=True)

        self.__pos_cash = pd.get_dummies(
            data=self.__pos_cash,
            dummy_na=True,
            columns=self.__pos_cash.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        pass

    def data_return(self):
        # print(self.__pos_cash.shape)
        self.__pos_cash.to_csv(os.path.join(self.__input_path, "pos_cash_temp.csv"), index=False)

        return self.__pos_cash


if __name__ == "__main__":
    ppc = PreparePosCash(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    ppc.data_prepare()
    ppc.data_transform()
    ppc.data_generate()
    ppc.data_return()
