import numpy as np
from numpy import int32, double
from plotly import plot
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from fbprophet import Prophet
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from datetime import datetime
from fbprophet.plot import add_changepoints_to_plot
import pyarrow as pa

dataFrameSerialization = "legacy"

start_time = datetime.now()
spark = SparkSession.builder.appName('testML').master("local[*]").getOrCreate()
result_schema = StructType([
    StructField('ds', TimestampType()),
    StructField('outletid', StringType()),
    StructField('OutletName', StringType()),
    StructField('y', IntegerType()),
    StructField('yhat', IntegerType()),
    StructField('yhat_upper', IntegerType()),
    StructField('yhat_lower', IntegerType()),
])

#confidence interval = 0.95 (95%)
#growth = linear
#daily_seasonality = true,.
#Future dataframe (forecast for next 24 hours)
#Periods = 48
#Frequency = 30mins

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def forecast_result(cost_pd):
    # define model parameter
    model = Prophet(interval_width=0.95, growth='linear', daily_seasonality=False)
    #model.add_seasonality(name="monthly", period=3, fourier_order=5)
    model.add_country_holidays(country_name='SG')
    # model itting
    model.fit(cost_pd)
    # make future dataframe with specified period and frequency
    future_pd = model.make_future_dataframe(periods=6, freq='M', include_history=True)
    # perform forecasting
    forecast_pd = model.predict(future_pd)
    #forecast_pd.plot()
    #pyplot.show()
    # convert negative values to zero
    num = forecast_pd._get_numeric_data()
    num[num < 0] = 0
    # join forecasted data with existing attributes
    f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
    cp_pd = cost_pd[['ds', 'outletid', 'y']].set_index('ds')
    result_pd = f_pd.join(cp_pd, how='left')
    result_pd.reset_index(level=0, inplace=True)
    result_pd['outletid'] = cost_pd['outletid'].iloc[0]
    result_pd['OutletName'] = cost_pd['OutletName'].iloc[0]
    print(result_pd[['ds', 'outletid', 'OutletName', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    model.plot(result_pd, xlabel="Outlet Name : " + cost_pd['OutletName'].iloc[0])
    pyplot.show()
    #model.plot_components(result_pd)

    return result_pd[['ds', 'outletid', 'OutletName', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) /np.abs(y_true))) * 100
    print('Mean absolute percentage error: ',mean_absolute_percentage_error(df.y,forecast_pd.yhat))

# load data into panda dataframe and convert ds column to date time
path = '/Users/dongxinyi/Desktop/prediction_data_bk.csv'
df = pd.read_csv(path)
df.info()
# summarize shape
print(df.shape)
# show first few rows
print(df.head())
#select date
df.columns = ['outletid','OutletName','ds','y']
# data conversion

df['ds'] = pd.Series(df['ds'])

df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# 0   2018-01-01
# 1          NaT
# dtype: datetime64[ns]

df['ds'] = df['ds'].dt.strftime('%Y-%m')

#df['ds'] = pd.to_datetime(df['ds'], errors='coerce').apply(lambda x: x.strftime('%Y-%m'))
df['outletid'] = df['outletid'].astype(str)
df['OutletName'] = df['OutletName'].astype(str)
df['y'] = df['y'].str.replace(',','',).astype(float)
# Get the count of each value
value_counts = df['outletid'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
to_remove = value_counts[value_counts < 2].index

# Keep rows where the city column is not in to_remove
df = df[~df['outletid'].isin(to_remove)]
df.dropna(subset=['ds'], inplace = True)
df = df.dropna(subset=['y'])
# plot the time series
df.plot()
pyplot.show()
print(df.head())
# Convert to Spark dataframe
sdf = spark.createDataFrame(df)
sdf.printSchema()
sdf.show(10)
sdf.count()
# Repartition dataframe by user id
costdf = sdf.repartition(spark.sparkContext.defaultParallelism, ['outletid']).cache()
# Apply time series forecasting
results = (costdf.groupby('outletid').apply(forecast_result).withColumn('training_date', current_date()))
# Visualize Some data
results.coalesce(1)
results.cache()
results.show()
# results.createOrReplaceTempView('forecasted')
results.write.option("header", "true").mode('overwrite').format('csv').save('./prediction_cost_next_month')
# print("total:", results.count(), "rows")
# results.createOrReplaceTempView('forecasted')
# spark.sql("SELECT car_park_no, count(*) FROM  forecasted GROUP BY car_park_no").show()
# final_df = results.toPandas()
#
# # display the chart
# final_df = final_df.set_index('ds')
# final_df.query('car_park_no == "A100"')[['y', 'yhat']].plot()
# plt.show()
#
# final_df.query('car_park_no == "A15"')[['y', 'yhat']].plot()
# plt.show()
print("total:", results.count(), "rows")
print('Duration: {}'.format(datetime.now() - start_time))
