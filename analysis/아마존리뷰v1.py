# Databricks notebook source
import pandas as pd
import numpy as np
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


# COMMAND ----------

# MAGIC %md
# MAGIC ## Amazon review 50000개

# COMMAND ----------

amazon_review = sc.textFile('dbfs:/FileStore/amazon/data/All_Amazon_Review_Sample50000.json/part-00000')
import json
ar_json = amazon_review.map(json.loads)
from pyspark.sql.types import StringType, StructType, StructField, FloatType, LongType, MapType, BooleanType

feature_schema = StructType([
    StructField("overall", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", LongType()),
    StructField("style", MapType(StringType(), StringType())),
    StructField("image", StringType())  
])

amazon_review_df = spark.createDataFrame(ar_json, feature_schema)
amazon_review_df.createOrReplaceTempView("amazon_review_df")
display(amazon_review_df)


# COMMAND ----------

amazon_review_df.printSchema()


# COMMAND ----------

amazon_review_df = amazon_review_df.select(
    col('overall'),col('vote'),col('verified'), col('reviewTime'), col('reviewerID'),col('asin'),
    col('reviewText'),col('summary') ,col('image'),col('style')
)

# COMMAND ----------

display(amazon_review_df)

# COMMAND ----------

amazon_review_pd = amazon_review_df.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC ###결측치 %

# COMMAND ----------

amazon_review_pd.isnull().sum() / len(amazon_review_pd) * 100


# COMMAND ----------

# MAGIC %md
# MAGIC ##

# COMMAND ----------

amazon_review_df.describe(['overall']).show()

# COMMAND ----------

amazon_review_df.describe(['vote']).show() # float로 바꿔줘야함

# COMMAND ----------

# MAGIC %md
# MAGIC ###vote 결측치 >> 0

# COMMAND ----------

amazon_review_df = amazon_review_df.na.fill({'vote': 0})
display(amazon_review_df)

# COMMAND ----------

amazon_review_df.describe(['vote']).show() 

# COMMAND ----------

amazon_review_df = amazon_review_df.withColumn("vote", col("vote").cast("float"))

# COMMAND ----------

amazon_review_df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### reviewtime to datetime

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth

amazon_review_df = amazon_review_df.withColumn("reviewTime", to_date("reviewTime", "MM d, yyyy")) \
                                   .withColumn("year", year("reviewTime")) \
                                   .withColumn("month", month("reviewTime")) \
                                   .withColumn("day", dayofmonth("reviewTime"))
display(amazon_review_df)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 누적리뷰수 (현재일기준 이전 누적 리뷰 개수)

# COMMAND ----------

from pyspark.sql.functions import col, to_date, count
from pyspark.sql.window import Window

review_data1 = amazon_review_df.select("asin", "reviewTime")
review_data1 = review_data1.withColumn("reviewTime", to_date(col("reviewTime")))
window_spec = Window.partitionBy("asin").orderBy("reviewTime")

# 하루 전 리뷰 개수를 계산
review_data1 = review_data1.withColumn("before_today_review_count", count("reviewTime").over(window_spec.rowsBetween(Window.unboundedPreceding, -1)))
display(review_data1)



# COMMAND ----------

#코드설명..
#over(window_spec.rowsBetween(Window.unboundedPreceding, -1)):
#over 함수는 윈도우 함수를 적용하기 위해 사용됩니다. 이 함수의 인자로는 윈도우 스펙을 지정합니다.
#window_spec는 윈도우 스펙을 정의하는데 사용되며, 여기서는 asin 열을 기준으로 파티션하고 reviewTime 열을 기준으로 정렬하여 윈도우를 설정합니다.
#rowsBetween(Window.unboundedPreceding, -1)은 윈도우 함수가 적용되는 범위를 정의합니다. 여기서는 현재 행 이전까지의 모든 행을 포함하여 윈도우를 #설정합니다.
#-------------------------------------------------------
#판다스 사용시
#review_data_pd['reviewTime'] = pd.to_datetime(review_data_pd['reviewTime'])
#review_data_pd['before_today_review_count'] = review_data_pd.groupby('asin')['reviewTime'].cumcount()


# COMMAND ----------

# style 열이 null이 아닌 값을 선택
notnull_style = amazon_review_df.filter(col("style").isNotNull())

# map 형식 보기 
notnull_style.select("style").show(truncate=False)



# COMMAND ----------

# MAGIC %md
# MAGIC ### sktyle : key , value 나눠서 별도의 컬럼으로 생성
# MAGIC #### ex ) {Color: ->  Black}" // style": {	"Size:": "Large", "Color:": "Charcoal"} 다른 카테고리에서는 이러한 형식으로도 존재하는거같음

# COMMAND ----------

# 각각의 키와 값을 별도의 컬럼으로
amazon_review_df = amazon_review_df.select(
    "*",
    col("style").getItem("key").alias("style_key"),
    col("style").getItem("value").alias("style_value")
)
display(amazon_review_df)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### key:color ,value: black 의 형태로 존재 , style 이 color에 대해서만 존재하면 meta data 의 Image 를 이용해 픽셀값을 이용해 채워 넣을수도 있을꺼 같음
# MAGIC
# MAGIC ##### 

# COMMAND ----------

# MAGIC %md
# MAGIC ###review image 존재 1 존재x 0 

# COMMAND ----------

from pyspark.sql.functions import when, col
amazon_review_df = amazon_review_df.withColumn("image_exist", when(col("image").isNull(), 0).otherwise(1))


# COMMAND ----------

display(amazon_review_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### review text, summary 의 경우 결측치 reviewText 0.082, summary 0.114, >> 제거할지 다른 값으로 채워 넣을지(중립적인언어?)고민 필요
# MAGIC ##### verified , image_exist 의 경우 추후에 원 핫 인코딩 필요

# COMMAND ----------


