# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("example-app") \
    .getOrCreate()

df = spark.read.csv("dbfs:/FileStore/amazon/data/1.csv", header=True, inferSchema=True)


# COMMAND ----------

display(df)

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T
import json


# PySpark UDF 정의
def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

# UDF 등록
retrieve_embedding = F.udf(parse_embedding_from_string, T.ArrayType(T.DoubleType()))

# 기존 데이터프레임 sdf의 embedding 열을 변환하여 embedding_new 열에 저장
df = df.withColumn("embedding_array", retrieve_embedding(F.col("embedding")))
# 원래의 embedding 열 삭제
df = df.drop("embedding")
# 스키마 데이터 출력
df.printSchema()


# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType

# 주어진 asin의 embedding_array 추출
target_embedding = df.filter(df['asin'] == '9791133239').select('embedding_array').collect()[0][0]

# NumPy 배열을 사용하지 않고 순수 Python 리스트를 활용하여 코사인 유사도 계산
def cosine_similarity_udf_func(vector1):
    target = list(target_embedding)
    vector1 = list(vector1)
    
    # dot product 계산
    dot_product = sum(p*q for p, q in zip(target, vector1))
    
    # norm 계산
    norm_target = sum(p**2 for p in target) ** 0.5
    norm_vector1 = sum(q**2 for q in vector1) ** 0.5
    
    # 코사인 유사도 계산
    norm_product = norm_target * norm_vector1
    return float(dot_product / norm_product) if norm_product != 0 else 0.0

# 코사인 유사도 계산을 위한 UDF 등록
cosine_similarity_udf = udf(cosine_similarity_udf_func, DoubleType())

# 코사인 유사도 계산
df = df.withColumn("cosine_similarity", cosine_similarity_udf(F.col("embedding_array")))

# 상위 10개의 유사한 제품 선택 (본인 제품은 제외)
top_10_similar_products = df.filter(df['asin'] != '9791133239').orderBy(F.col("cosine_similarity").desc()).limit(10)

# top 10 asin 및 imageURLHighRes를 출력
top_10_similar_products.select("asin", "imageURLHighRes").show()


# COMMAND ----------

# top 10 제품의 asin, imageURLHighRes, 코사인 유사도를 출력
display(top_10_similar_products.select("asin", "imageURLHighRes", "cosine_similarity"))

# COMMAND ----------

# 하위 10개의  제품 
bottom_10_similar_products = df.filter(df['asin'] != '9791133239').orderBy(F.col("cosine_similarity").asc()).limit(10)

display(bottom_10_similar_products.select("asin", "imageURLHighRes", "cosine_similarity"))

# COMMAND ----------

# top 10 유사 제품의 asin, imageURLHighRes 및 코사인 유사도를 출력
display(top_10_similar_products.select("asin", "imageURLHighRes", "cosine_similarity"))


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("cosine_similarity_example").getOrCreate()

# 코사인 유사도를 함수
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p, q in zip(vector1, vector2))
    
    norm_vector1 = sum(p**2 for p in vector1) ** 0.5
    norm_vector2 = sum(q**2 for q in vector2) ** 0.5
    
    # 코사인 유사도 계산
    norm_product = norm_vector1 * norm_vector2
    return float(dot_product / norm_product) if norm_product != 0 else 0.0

# udf
cosine_similarity_udf = udf(cosine_similarity, DoubleType())

# 코사인 유사도 추가
df = df.withColumn("cos_similarity", cosine_similarity_udf(col("asin1_embedding"), col("asin2_embedding")))

display(df)

# COMMAND ----------


