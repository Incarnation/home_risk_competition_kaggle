# coding:utf-8

import os
import re
import sys
import importlib
import numpy as np
import featuretools as ft
import competition_utils as cu
from featuretools.primitives import Sum
from featuretools.primitives import Std
from featuretools.primitives import Max
from featuretools.primitives import Min
from featuretools.primitives import Median
from featuretools.primitives import Count
from featuretools.primitives import PercentTrue
from featuretools.primitives import Trend
from featuretools.primitives import AvgTimeBetween
# from dask.distributed import LocalCluster


class FeatureToolsTrainV7(object):
    def __init__(self, input_path, output_path):
        # init
        self.__input_path, self.__output_path = input_path, output_path
        self.__prepare_application_train = importlib.import_module("PrepareApplicationTrain")
        self.__prepare_bureau = importlib.import_module("PrepareBureau")
        self.__prepare_bureau_balance = importlib.import_module("PrepareBureauBalance")
        self.__prepare_previous_application = importlib.import_module("PreparePreviousApplication")
        self.__prepare_credit_card = importlib.import_module("PrepareCreditCard")
        self.__prepare_installment_payment = importlib.import_module("PrepareInstallmentPayment")
        self.__prepare_pos_cash = importlib.import_module("PreparePosCash")

        # data prepare
        self.__application_train_df = None
        self.__bureau_df = None
        self.__bureau_balance_df = None
        self.__previous_application_df = None
        self.__credit_card_df = None
        self.__installment_payment_df = None
        self.__pos_cash_df = None

        # es set
        self.__es = None
        self.__train_feature_matrix = None
        self.__train_feature = None
        self.__application_train_df_variable_types = dict()
        self.__bureau_df_variable_types = dict()
        self.__bureau_balance_df_variable_types = dict()
        self.__previous_application_df_variable_types = dict()
        self.__credit_card_df_variable_types = dict()
        self.__installment_payment_df_variable_types = dict()
        self.__pos_cash_df_variable_types = dict()

        # run dfs
        # self.__cluster = None

    def data_prepare(self):
        print("data prepare for application train start:")
        pat = self.__prepare_application_train.PrepareApplicationTrain(input_path=self.__input_path)
        pat.data_prepare()
        pat.data_transform()
        pat.data_generate()
        self.__application_train_df = pat.data_return()
        print("data prepare for application train done:")
        # 这里针对分类变量的处理有不足之处
        # 静态信息表 1、原始分类变量, 之后使用 Target Encoder 2、使用类别占比 Encoder
        # 流水信息表 1、原始分类变量, NUM_UNIQUE、MODE 2、Dummy 分类变量, 当做布尔值处理

        print("data prepare for bureau data start:")
        pb = self.__prepare_bureau.PrepareBureau(input_path=self.__input_path)
        pb.data_prepare()
        pb.data_transform()
        pb.data_generate()
        self.__bureau_df = pb.data_return()
        print("data prepare for bureau data done:")

        print("data prepare for bureau balanace data start:")
        pbb = self.__prepare_bureau_balance.PrepareBureauBalance(input_path=self.__input_path)
        pbb.data_prepare()
        pbb.data_transform()
        pbb.data_generate()
        self.__bureau_balance_df = pbb.data_return()
        print("data prepare for bureau balanace data done:")

        print("data prepare for previous application data start:")
        ppa = self.__prepare_previous_application.PreparePreviousApplication(input_path=self.__input_path)
        ppa.data_prepare()
        ppa.data_transform()
        ppa.data_generate()
        self.__previous_application_df = ppa.data_return()
        print("data prepare for previous application data done:")

        print("data prepare for credit data start:")
        pcc = self.__prepare_credit_card.PrepareCreditCard(input_path=self.__input_path)
        pcc.data_prepare()
        pcc.data_transform()
        pcc.data_generate()
        self.__credit_card_df = pcc.data_return()
        print("data prepare for credit data done:")

        print("data prepare for installment payment data start:")
        pip = self.__prepare_installment_payment.PrepareInstallmentPayment(input_path=self.__input_path)
        pip.data_prepare()
        pip.data_transform()
        pip.data_generate()
        self.__installment_payment_df = pip.data_return()
        print("data prepare for installment payment data done:")

        print("data prepare for posh cash data start:")
        ppc = self.__prepare_pos_cash.PreparePosCash(input_path=self.__input_path)
        ppc.data_prepare()
        ppc.data_transform()
        ppc.data_generate()
        self.__pos_cash_df = ppc.data_return()
        print("data prepare for posh cash data done:")

        self.__application_train_df["SK_ID_CURR"] = self.__application_train_df["SK_ID_CURR"].astype(np.int64)
        self.__bureau_df["SK_ID_CURR"] = self.__bureau_df["SK_ID_CURR"].astype(np.int64)
        self.__bureau_df["SK_ID_BUREAU"] = self.__bureau_df["SK_ID_BUREAU"].astype(np.int64)
        self.__bureau_balance_df["SK_ID_BUREAU"] = self.__bureau_balance_df["SK_ID_BUREAU"].astype(np.int64)
        self.__previous_application_df["SK_ID_CURR"] = self.__previous_application_df["SK_ID_CURR"].astype(np.int64)
        self.__previous_application_df["SK_ID_PREV"] = self.__previous_application_df["SK_ID_PREV"].astype(np.int64)
        self.__credit_card_df["SK_ID_PREV"] = self.__credit_card_df["SK_ID_PREV"].astype(np.int64)
        self.__installment_payment_df["SK_ID_PREV"] = self.__installment_payment_df["SK_ID_PREV"].astype(np.int64)
        self.__pos_cash_df["SK_ID_PREV"] = self.__pos_cash_df["SK_ID_PREV"].astype(np.int64)

    def es_set(self):
        for col in self.__application_train_df.select_dtypes("object").columns.tolist():
            self.__application_train_df_variable_types[col] = ft.variable_types.Categorical

        for col in self.__bureau_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__bureau_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__bureau_balance_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__bureau_balance_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__previous_application_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__previous_application_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__credit_card_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__credit_card_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__installment_payment_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__installment_payment_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__pos_cash_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__pos_cash_df_variable_types[col] = ft.variable_types.Boolean

        self.__es = ft.EntitySet(id="application_train")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="application_train",
            dataframe=self.__application_train_df,
            index="SK_ID_CURR",
            variable_types=self.__application_train_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau",
            dataframe=self.__bureau_df,
            index="SK_ID_BUREAU",
            time_index="TIME_DAYS_CREDIT",
            variable_types=self.__bureau_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau_balance",
            dataframe=self.__bureau_balance_df,
            make_index=True,
            index="bureau_balance_id",
            time_index="TIME_MONTHS_BALANCE",
            variable_types=self.__bureau_balance_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="previous_application",
            dataframe=self.__previous_application_df,
            index="SK_ID_PREV",
            time_index="TIME_DAYS_DECISION",
            variable_types=self.__previous_application_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="credit_card",
            dataframe=self.__credit_card_df,
            make_index=True,
            index="credit_card_id",
            time_index="TIME_MONTHS_BALANCE",
            variable_types=self.__credit_card_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="installment_payment",
            dataframe=self.__installment_payment_df,
            make_index=True,
            index="installment_payment_id",
            time_index="TIME_DAYS_INSTALMENT",
            variable_types=self.__installment_payment_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="pos_cash",
            dataframe=self.__pos_cash_df,
            make_index=True,
            index="pos_cash_id",
            time_index="TIME_MONTHS_BALANCE",
            variable_types=self.__pos_cash_df_variable_types
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["application_train"]["SK_ID_CURR"],
                self.__es["bureau"]["SK_ID_CURR"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["bureau"]["SK_ID_BUREAU"],
                self.__es["bureau_balance"]["SK_ID_BUREAU"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["application_train"]["SK_ID_CURR"],
                self.__es["previous_application"]["SK_ID_CURR"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["pos_cash"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["credit_card"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["installment_payment"]["SK_ID_PREV"]
            )
        )
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Refused"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Approved"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_PRODUCT_TYPE_walk-in"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_CODE_REJECT_REASON_HC"].interesting_values = [1]
        self.__es["bureau"]["FLAG_BUREAU_CREDIT_ACTIVE_Active"].interesting_values = [1]
        self.__es["bureau"]["FLAG_BUREAU_CREDIT_ACTIVE_Closed"].interesting_values = [1]
        print(self.__es)

    def dfs_run(self):
        # self.__cluster = LocalCluster(
        #     n_workers=2,
        #     memory_limit="20g"
        # )

        # self.__train_feature_matrix, self.__train_feature = ft.dfs(
        #     entityset=self.__es,
        #     target_entity="application_train",
        #     agg_primitives=[Sum, Std, Max, Min, Median, Count, PercentTrue, Trend, AvgTimeBetween],
        #     trans_primitives=[],
        #     where_primitives=[Std, Max, Min, Median, Count],
        #     dask_kwargs={"cluster": self.__cluster},
        #     verbose=True
        # )

        self.__train_feature_matrix, self.__train_feature = ft.dfs(
            entityset=self.__es,
            target_entity="application_train",
            agg_primitives=[Sum, Std, Max, Min, Median, Count, PercentTrue, Trend, AvgTimeBetween],
            trans_primitives=[],
            where_primitives=[Std, Max, Min, Median, Count],
            verbose=True,
            chunk_size=110,
            max_depth=2
        )

        self.__train_feature_matrix = cu.remove_li_features(self.__train_feature_matrix)
        print(f'{len(self.__train_feature)} Total Features created')
        print(self.__train_feature_matrix.head())
        self.__train_feature_matrix.to_csv(os.path.join(self.__output_path, "train_feature_df.csv"), index=False)

        # ft.save_features(self.__train_feature, os.path.join(self.__output_path, "train_feature_definitions"))
        # self.__train_feature = ft.load_features("train_feature_definitions")

        # self.__train_feature_matrix = ft.calculate_feature_matrix(
        #     features=self.__train_feature,
        #     entityset=self.__es,
        #     n_jobs=2
        # )
        #
        # self.__train_feature_matrix.to_csv(os.path.join(self.__output_path, "train_feature_df.csv"), index=True)

if __name__ == "__main__":
    ftt = FeatureToolsTrainV7(
        input_path="D:\Kaggle\Home_Credit_Default_Risk\clean_data",
        output_path="D:\Kaggle\Home_Credit_Default_Risk\clean_data"
    )
    ftt.data_prepare()
    ftt.es_set()
    ftt.dfs_run()
