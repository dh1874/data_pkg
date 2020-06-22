from fbprophet import Prophet
import pandas as pd
import time


class NormProphet(object):
    """docstring for NormProphet"""
    def __init__(self):
        super(NormProphet, self).__init__()
        # 模型对象
        self.model_days = None
        # 节假日数据DataFrame
        self.holidays = None

        # 模型参数
        self.weekly_seasonality = 3
        self.weekly_prior_scale = 10
        self.monthly_seasonality = 20
        self.monthly_prior_scale = 10
        self.seasonality = 10
        self.seasonality_prior_scale = 10
        self.yearly_seasonality = 10
        self.yearly_prior_scale = 10
        self.holidays_seasonality = 10
        self.holidays_prior_scale = 20
        self.changepoint_prior_scale = 0.2

        # 訓練集的開始結束時間, str
        self.train_end_time: str = '0'
        self.train_start_time: str = '0'

        # 預測結果保留列
        self.result_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']

    def predict_future(self, x_data, predict_start_time, predict_end_time):

        self.__get_holidays()

        # 训练集内的开始结束时间
        self.train_start_time = str(x_data['ds'].min()).split(" ")[0]
        self.train_end_time = str(x_data['ds'].max()).split(" ")[0]

        print(self.train_end_time, type(self.train_end_time))

        #  初始化模型
        self.model_days = Prophet(
            holidays=self.holidays,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            # yearly_seasonality=int(self.yearly_seasonality),
            # weekly_seasonality=int(self.weekly_seasonality),
            seasonality_mode="multiplicative",
            # growth='logistic',
            changepoint_prior_scale=self.changepoint_prior_scale
        )

        # 模型参数，周期性
        self.model_days.add_seasonality(name='monthly',
                                        period=30.5,
                                        fourier_order=int(self.monthly_seasonality),
                                        prior_scale=self.monthly_prior_scale)

        # 模型参数，节假日
        self.model_days.add_country_holidays(country_name='CN')

        # self.model_days.seasonalities['weekly']['fourier_order'] \
        #     = int(self.weekly_seasonality)
        #
        # self.model_days.seasonalities['yearly']['fourier_order'] \
        #     = int(self.yearly_seasonality)

        # 训练
        self.model_days.fit(x_data)

        # 得到训练结束日期 - 预测结束日之间的天数
        periods = self.__calculate_period(predict_end_time)

        # 天预测
        # 得到待预测的future，DataFrame['ds']
        future = self.model_days.make_future_dataframe(periods)

        # 完整的预测结果
        days_forecast = self.model_days.predict(future)

        # 行列，列过滤
        days_forecast = days_forecast[days_forecast['ds'] >= predict_start_time][self.result_cols]

        return days_forecast

    def __calculate_period(self, predict_end_time):
        """
        计算训练结束日 至 预测结束日之间的间隔天数
        """
        begin_time = time.mktime(time.strptime(self.train_end_time, "%Y-%m-%d"))
        end_time = time.mktime(time.strptime(predict_end_time,"%Y-%m-%d"))
        timestamp = end_time - begin_time
        periods = int(timestamp // (86400) + 1)
        return periods

    def __get_holidays(self):
        """
        获取指定的节假日DataFrame
        """
        spring_festivals = pd.DataFrame({
            'holiday': 'spring_festival',
            'ds': pd.to_datetime(['2014-01-31', '2015-02-19',
                                  '2016-02-08', '2017-01-28',
                                  '2018-02-16', '2019-02-05',
                                  '2020-01-25', '2021-02-12',
                                  '2022-02-01', '2023-01-22',
                                  '2024-02-10', '2025-01-29']),
            'lower_window': -15,
            'upper_window': 15,
        })

        national_days = pd.DataFrame({
            'holiday': 'national_day',
            'ds': pd.to_datetime(['2014-10-01', '2015-10-01',
                                  '2016-10-01', '2017-10-01',
                                  '2018-10-01', '2019-10-01',
                                  '2020-10-01', '2021-10-01',
                                  '2022-10-01', '2023-10-01',
                                  '2024-10-01', '2025-10-01']),
            'lower_window': 0,
            'upper_window': 6,
        })

        double_eleven = pd.DataFrame({
            'holiday': 'double_eleven',
            'ds': pd.to_datetime(['2014-11-11', '2015-11-11',
                                  '2016-11-11', '2017-11-11',
                                  '2018-11-11', '2019-11-11',
                                  '2020-11-11', '2021-11-11',
                                  '2022-11-11', '2023-11-11',
                                  '2024-11-11', '2025-11-11']),
            'lower_window': -7,
            'upper_window': 7,
        })

        self.holidays = pd.concat([spring_festivals,
                                   national_days,
                                   double_eleven])