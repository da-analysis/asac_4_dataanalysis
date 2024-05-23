# Databricks notebook source
# MAGIC %pip install sparknlp 

# COMMAND ----------

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings, AlbertEmbeddings, Tokenizer, Normalizer, StopWordsCleaner, RoBertaSentenceEmbeddings, Doc2VecModel
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.functions import col, size, array, expr
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

# COMMAND ----------

# 이미지
# df_image =  spark.read.csv("dbfs:/FileStore/amazon/data/image/AC_image_embedding",header=True)


# 텍스트
df_text = spark.read.table("asac.embed_cell_sbert_32_fin")

# 5배수
df_5 = spark.read.table("asac.240430_review3_5multiply")

# 15배수
df_15 = spark.read.table("asac.240430_review3_15multiply")

# 20배수
df_20 = spark.read.table("asac.240430_review3_20multiply")


# COMMAND ----------

## 완전 전체 train 셋과 test 셋
total_train = spark.read.table("asac.240430_train_df")
total_test =  spark.read.table("asac.240430_test_df")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.embed_cell_sbert_32_fin

# COMMAND ----------

# MAGIC %md
# MAGIC  asin별로 text 임베딩 값 개수 약 59만개

# COMMAND ----------

df_5.count()

# COMMAND ----------

display(df_text)

# COMMAND ----------

display(df_5)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 이미지 임베딩 전처리
# MAGIC - array로 변환

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import json

df_image =  spark.read.csv("dbfs:/FileStore/amazon/data/image/AC_image_embedding",header=True)
df_image = df_image.drop("_c0")

# PySpark UDF 정의
def parse_embedding_from_string(x):
    res = json.loads(x)
    return res

# UDF 등록
retrieve_embedding = F.udf(parse_embedding_from_string, T.ArrayType(T.DoubleType()))

df_image = df_image.withColumn("embedding_array", retrieve_embedding(F.col("embedding")))

# 원래의 embedding 열 삭제
df_image = df_image.drop("embedding")
# nan 값 있는 것 제거
df_image = df_image.dropna(subset=["embedding_array"])

# 스키마 데이터 출력
df_image.printSchema()

# COMMAND ----------

display(df_image.limit(10))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## df_5 컬럼 합치기

# COMMAND ----------

# MAGIC %md
# MAGIC #### 텍스트 합치기

# COMMAND ----------

col_names = df_5.columns
col_names

# COMMAND ----------

# asin1과 asin2 를 기준으로 left outer join 진행
# 임베딩, 통계량 값, 길이

df_5 = df_5.join(df_text, df_5.asin1 == df_text.asin,"left_outer")
df_5 = df_5.drop("asin")

for col_name in df_5.columns:
    if col_name not in col_names:  
        df_5 = df_5.withColumnRenamed(col_name, col_name + "_1")

df_text_renamed = df_text.select(['asin'] + [col(c).alias(c + '_2') for c in df_text.columns if c != 'asin'])

# df_5 변경된 df_text_renamed 조인
df_5 = df_5.join(df_text_renamed, df_5.asin2 == df_text_renamed.asin)

# 필요하지 않은 df_text asin 컬럼 삭제
df_5 = df_5.drop(df_text_renamed.asin)

df_5 = df_5.drop("asin")
df_5 = df_5.drop("variance_1")
df_5 = df_5.drop("variance_2")

# COMMAND ----------

display(df_5.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 이미지 합치기

# COMMAND ----------

from pyspark.sql.functions import col

df_5 = df_5.join(df_image.alias("image1"), df_5.asin1 == col("image1.asin"), "left_outer")
df_5 = df_5.withColumnRenamed("embedding_array", "image_1")
df_5 = df_5.drop("asin")

df_5 = df_5.join(df_image.alias("image2"), df_5.asin2 == col("image2.asin"), "left_outer")
df_5 = df_5.withColumnRenamed("embedding_array", "image_2")
df_5 = df_5.drop("asin")

# COMMAND ----------

df_5_cos = df_5

# COMMAND ----------

# MAGIC %md
# MAGIC #### 텍스트 임베딩, 이미지임베딩 null 값 확인하기
# MAGIC - df_5 원래 103752개
# MAGIC - 텍스트는 new_pcaValues32_1, new_pcaValues32_2
# MAGIC - 이미지는 image_1, image_1 확인하기

# COMMAND ----------

new_pcaValues32_1 = df_5_cos.filter(df_5_cos["new_pcaValues32_1"].isNull()).count()
new_pcaValues32_2 = df_5_cos.filter(df_5_cos["new_pcaValues32_2"].isNull()).count()
image_1 = df_5_cos.filter(df_5_cos["image_1"].isNull()).count()
image_2 = df_5_cos.filter(df_5_cos["image_2"].isNull()).count()

print(f"'new_pcaValues32_1' 컬럼의 널 값 개수: {new_pcaValues32_1}")
print(f"'new_pcaValues32_2' 컬럼의 널 값 개수: {new_pcaValues32_2}")
print(f"'image_1' 컬럼의 널 값 개수: {image_1}")
print(f"'image_2' 컬럼의 널 값 개수: {image_2}")

# COMMAND ----------

display(df_5_cos.filter(df_5_cos["new_pcaValues32_1"].isNull()))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 코사인 유사도 계산 진행 (방법3으로 해서 진행해보기)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 방법3) udf 안쓰고 활용하는 방법

# COMMAND ----------

from pyspark.sql.functions import expr, col, when, sqrt

columns = [
    ("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"), ("q1_1", "q1_2"),
    ("q2_1", "q2_2"), ("q3_1", "q3_2"), ("skewness_1", "skewness_2"),
    ("kurtosis_1", "kurtosis_2")
]

for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(32)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(32)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(32)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # null 조건 체크 후 코사인 유사도 계산 또는 null 할당
    cosine_similarity_condition = when(
        col(col1).isNull() | col(col2).isNull(), 
        None
    ).otherwise(expr(cosine_similarity_expr))
    
    df_5_cos_limit10 = df_5_cos.limit(10)
    df_5_cos_limit10 = df_5_cos_limit10.withColumn(f"{col1[:-2]}_cosine_similarity", cosine_similarity_condition)
    df_5_cos = df_5_cos.withColumn(f"{col1[:-2]}_cosine_similarity", cosine_similarity_condition)

# 최종 코사인 유사도 평균 계산 진행
df_5_cos_limit10 = df_5_cos_limit10.withColumn("cosine_fin", (
    col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") +
    col("q1_cosine_similarity") + col("q2_cosine_similarity") +
    col("q3_cosine_similarity") + col("skewness_cosine_similarity") + col("kurtosis_cosine_similarity")
) / 7)

df_5_cos = df_5_cos.withColumn("cosine_fin", (
    col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") +
    col("q1_cosine_similarity") + col("q2_cosine_similarity") +
    col("q3_cosine_similarity") + col("skewness_cosine_similarity") + col("kurtosis_cosine_similarity")
) / 7)


# COMMAND ----------

from pyspark.sql.functions import expr, col, when, sqrt

columns = [
    ("image_1", "image_2")
]

for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(1024)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(1024)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(1024)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # null 조건 체크 후 코사인 유사도 계산 또는 null 할당
    cosine_similarity_condition = when(
        col(col1).isNull() | col(col2).isNull(), 
        None
    ).otherwise(expr(cosine_similarity_expr))
    
    df_5_cos_limit10 = df_5_cos.limit(10)
    df_5_cos_limit10 = df_5_cos_limit10.withColumn(f"{col1[:-2]}_cosine_similarity", cosine_similarity_condition)
    df_5_cos = df_5_cos.withColumn(f"{col1[:-2]}_cosine_similarity", cosine_similarity_condition)


# COMMAND ----------

display(df_5_cos_limit10)

# COMMAND ----------

from pyspark.sql.functions import expr, col
from pyspark.sql.functions import col, sqrt, sum as _sum, when
columns = [
    ("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"),("q1_1", "q1_2"),
    ("q2_1", "q2_2"),("q3_1", "q3_2"),("skewness_1", "skewness_2"),("kurtosis_1", "kurtosis_2")
]

# 각 컬럼 쌍에 대해 반복
for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(32)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(32)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(32)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # DataFrame에 코사인 유사도 컬럼 추가
    df_5_cos = df_5_cos.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    df_5_cos = df_5_cos.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
df_5_cos = df_5_cos.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

from pyspark.sql.functions import expr, col
from pyspark.sql.functions import col, sqrt, sum as _sum, when
columns = [
    ("image_1", "image_2")
]

# 각 컬럼 쌍에 대해 반복
for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(1024)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(1024)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(1024)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # DataFrame에 코사인 유사도 컬럼 추가
    df_5_cos = df_5_cos.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    df_5_cos = df_5_cos.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
df_5_cos = df_5_cos.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 방법4) pandas udf 활용

# COMMAND ----------

# 코사인 유사도 계산 컬럼 저장 (일단 이미지 빼고 저장하기)
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import numpy as np

columns = [("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"),("q1_1", "q1_2"),
    ("q2_1", "q2_2"),("q3_1", "q3_2"),("skewness_1", "skewness_2"),("kurtosis_1", "kurtosis_2")]

@pandas_udf("double", PandasUDFType.SCALAR)
def cosine_similarity_udf(v1: pd.Series, v2: pd.Series) -> pd.Series:
    # 각 Series의 요소가 벡터인 경우를 처리하기 위한 수정
    dot_product = np.array([np.dot(a, b) for a, b in zip(v1, v2)])
    norm_v1 = np.sqrt(np.array([np.dot(a, a) for a in v1]))
    norm_v2 = np.sqrt(np.array([np.dot(b, b) for b in v2]))
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return pd.Series(cosine_similarity)


# DataFrame에 코사인 유사도 컬럼 추가
for col1,col2 in columns:
    df_5_cos = df_5_cos.withColumn(f"{col1[:-2]}_cosine_similarity",  cosine_similarity_udf(col(col1), col(col2)))
    df_5_cos = df_5_cos.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# 최종 코사인 유사도 평균 계산 (텍스트만)
df_5_cos = df_5_cos.withColumn("text_cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 테이블 저장하기

# COMMAND ----------

name = "asac.df_5_cos"
df_5_cos.write.saveAsTable(name)

# COMMAND ----------

df_5_cos = spark.read.table("asac.df_5_cos")

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin1, image_1, image_2, image_cosine_similarity from asac.df_5_cos
# MAGIC limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.df_5_cos

# COMMAND ----------

# path1 = "dbfs:/FileStore/amazon/model/df_cos/"
# df_5_cos.to_parquet('%s/df_5_cos.parquet' % path1, compression='zstd')

# path1 = "dbfs:/FileStore/amazon/model/df_cos/"
# df_5_cos.write.format('parquet').option('compression', 'zstd').save('%s/df_5_cos.parquet' % path1)

# COMMAND ----------



# COMMAND ----------

df_5_cos.columns

# COMMAND ----------



# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC ## 부스팅 모델링 (Gradient-Boosted Tree (GBT))
# MAGIC 1. 텍스트 통계량 유사도 여러개만 추가한 모델
# MAGIC 2. 텍스트 통계량 임베딩만 추가한 모델
# MAGIC 3. 텍스트 임베딩만 추가한 모델
# MAGIC 4. 이미지 유사도만 추가한 모델
# MAGIC 5. 이미지 임베딩만 추가한 모델
# MAGIC 6. 1&4 - 텍스트 유사도 - 이미지 유사도만
# MAGIC 7. 2&5 - 통계량 임베딩- 임베딩만 
# MAGIC 8. 3&5 - 텍스트 임베딩 - 이미지 임베딩 
# MAGIC 9. 텍스트 평균 유사도 & 이미지 유사도
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# 데이터 준비 및 전처리
vectorAssembler_1 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity'], outputCol="features")


vectorAssembler_2 = VectorAssembler(inputCols=['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2',
 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2' ], outputCol="features")

vectorAssembler_3 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2' ], outputCol="features")

vectorAssembler_4 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'image_cosine_similarity',], outputCol="features")

vectorAssembler_5 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'image_1', 'image_2'], outputCol="features")

vectorAssembler_6 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity',
 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity',
 'kurtosis_cosine_similarity', 'image_cosine_similarity'], outputCol="features")

vectorAssembler_7 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2',
 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2' ], outputCol="features")

vectorAssembler_8 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec',
 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2'
 ], outputCol="features")

vectorAssembler_9 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity',
 'cosine_fin' ], outputCol="features")

# COMMAND ----------

new_pcaValues32_1 = df_5_cos.filter(df_5_cos["new_pcaValues32_1"].isNull()).count()
new_pcaValues32_2 = df_5_cos.filter(df_5_cos["new_pcaValues32_2"].isNull()).count()
image_1 = df_5_cos.filter(df_5_cos["image_1"].isNull()).count()
image_2 = df_5_cos.filter(df_5_cos["image_2"].isNull()).count()

print(f"'new_pcaValues32_1' 컬럼의 널 값 개수: {new_pcaValues32_1}")
print(f"'new_pcaValues32_2' 컬럼의 널 값 개수: {new_pcaValues32_2}")
print(f"'image_1' 컬럼의 널 값 개수: {image_1}")
print(f"'image_2' 컬럼의 널 값 개수: {image_2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1번 모델링 : 텍스트 통계량 유사도 여러개만 추가한 모델
# MAGIC

# COMMAND ----------

# 텍스트 없는 null 값가진 행제거한 데이터 셋
df_5_cos_null_text = df_5_cos.na.drop(subset=["new_pcaValues32_1"])
df_5_cos_null_text = df_5_cos_null_text.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'target', 'new_pcaValues32_1', 'list_length_1', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'new_pcaValues32_2', 'list_length_2', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2',
 'skewness_2', 'kurtosis_2', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity',
 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity',
 'kurtosis_cosine_similarity',  'cosine_fin'])

# train 데이터와 test 데이터 나누기
fractions = df_5_cos_null_text.select("target").distinct().rdd.flatMap(lambda x: x).collect()
fractions = {row: 0.8 for row in fractions}  # 트레인셋 80%

# `sampleBy` 함수를 사용하여 트레인셋 추출
train_df = df_5_cos_null_text.sampleBy("target", fractions, seed=42)

# `exceptAll`을 이용해서 트레인셋에 없는 행들을 테스트셋으로 설정
test_df = df_5_cos_null_text.exceptAll(train_df)

# 결과 확인
print(f"Total count: {df_5_cos_null_text.count()}")
print(f"Train count: {train_df.count()}, Test count: {test_df.count()}")

# 여기서 추가로 std, 왜도, 첨도 na 삭제했음

# COMMAND ----------

from pyspark.sql.functions import col, count, lit
# target 값의 개수 계산
target_counts = train_df.groupBy("target").agg(count("target").alias("count"))

# 전체 행의 수 계산
total_rows = train_df.count()

# 비율 계산
target_ratio = target_counts.withColumn("ratio", col("count") / lit(total_rows))

# 결과 출력
target_ratio.show()

# COMMAND ----------

# target 값의 개수 계산
target_counts = test_df.groupBy("target").agg(count("target").alias("count"))

# 전체 행의 수 계산
total_rows = test_df.count()

# 비율 계산
target_ratio = target_counts.withColumn("ratio", col("count") / lit(total_rows))

# 결과 출력
target_ratio.show()

# COMMAND ----------

df_5_cos_null_text.columns

# COMMAND ----------

display(df_5_cos_null_text.select('std_dev_cosine_similarity'))

# COMMAND ----------

# GBT 모델 설정 및 학습
vectorAssembler_1 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity'], outputCol="features")

train_df_1 = train_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()

gbt = GBTClassifier(labelCol="target", featuresCol="features", maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_1, gbt])
model = pipeline.fit(train_df_1)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
test_df_1 = test_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
test_df_1 = test_df_1.na.drop()
predictions_1 = model.transform(test_df_1)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_1)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_1)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_1)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")
precision = evaluatorPrecision.evaluate(predictions_1)
recall = evaluatorRecall.evaluate(predictions_1)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

from pyspark.sql.functions import col, count, lit
# target 값의 개수 계산
target_counts = train_df_1.groupBy("target").agg(count("target").alias("count"))

# 전체 행의 수 계산
total_rows = train_df_1.count()

# 비율 계산
target_ratio = target_counts.withColumn("ratio", col("count") / lit(total_rows))

# 결과 출력
target_ratio.show()

# COMMAND ----------

from pyspark.sql.functions import col, count, lit
# target 값의 개수 계산
target_counts = test_df_1.groupBy("target").agg(count("target").alias("count"))

# 전체 행의 수 계산
total_rows = test_df_1.count()

# 비율 계산
target_ratio = target_counts.withColumn("ratio", col("count") / lit(total_rows))

# 결과 출력
target_ratio.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1번 방법 가중치 주고 나서 한 것

# COMMAND ----------

from pyspark.sql.functions import when
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline

# 불균형 가중치 
numPositives = train_df_1.filter(train_df_1["target"] == 1).count()
numNegatives = train_df_1.filter(train_df_1["target"] == 0).count()
total = train_df_1.count()

balanceRatio = numNegatives / total

train_df_1 = train_df_1.withColumn('classWeight', when(train_df_1['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_1, gbt])
model_1_we = pipeline.fit(train_df_1)

predictions_1_we = model_1_we.transform(test_df_1)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_1_we)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_1_we)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_1_we)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_1_we)
recall = evaluatorRecall.evaluate(predictions_1_we)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

# 모델 저장
model_path = "dbfs:/FileStore/amazon/model/model"
model.write().overwrite().save(model_path)

model_path_1_we = "dbfs:/FileStore/amazon/model/model_1_we"
model_1_we.write().overwrite().save(model_path_1_we)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2번 모델링 : 텍스트 통계량 임베딩만 추가한 모델

# COMMAND ----------

train_df_2 = train_df.select(['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2','target'])
train_df_2 = train_df_2.na.drop()
test_df_2 = test_df.select(['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth',
 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity',
 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2','target'])
test_df_2 = test_df_2.na.drop()

vectorAssembler_2 = VectorAssembler(inputCols=['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2' ], outputCol="features")

# 임베딩 변수 컬럼으로 변환하는 것 필요함
# 배열 컬럼 이름 리스트
array_columns = ['mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2']

# 새로운 컬럼과 이에 해당하는 원래 컬럼의 매핑 딕셔너리
new_columns = {}

for array_col in array_columns:
    # 32개 원소에 대해 각각 반복
    for i in range(32):
        # 새 컬럼 이름: 원래 배열 컬럼 이름 + "_" + 원소 순서(index)
        new_col_name = f"{array_col}_{i+1}"
        # 새 컬럼 딕셔너리에 추가
        new_columns[new_col_name] = array_col
    
# 각 배열 요소를 새로운 컬럼으로 변환
for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
    i = int(new_col_name.split("_")[-1]) - 1
    train_df_2 = train_df_2.withColumn(new_col_name, col(array_col)[i])

for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
    i = int(new_col_name.split("_")[-1]) - 1
    test_df_2 = test_df_2.withColumn(new_col_name, col(array_col)[i])

columns_to_drop = ['mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2']
train_df_2 = train_df_2.drop(*columns_to_drop)
test_df_2 = test_df_2.drop(*columns_to_drop)


from pyspark.ml.feature import VectorAssembler

# 새롭게 생성된 컬럼들을 포함한 전체 컬럼 리스트 생성
assembler_columns = ['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity'] + list(new_columns.keys())

# VectorAssembler 인스턴스 생성
vectorAssembler_2 = VectorAssembler(inputCols=assembler_columns, outputCol="features")

# COMMAND ----------

# 불균형 가중치 
numPositives = train_df_2.filter(train_df_2["target"] == 1).count()
numNegatives = train_df_2.filter(train_df_2["target"] == 0).count()
total = train_df_2.count()

balanceRatio = numNegatives / total

train_df_2 = train_df_2.withColumn('classWeight', when(train_df_2['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_2, gbt])
model_2 = pipeline.fit(train_df_2)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_2_w = model_2.transform(test_df_2)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_2_w)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_2_w)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_2_w)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")
precision = evaluatorPrecision.evaluate(predictions_2_w)
recall = evaluatorRecall.evaluate(predictions_2_w)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_2 = "dbfs:/FileStore/amazon/model/model_2"
model_2.write().overwrite().save(model_path_2)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3번 모델링 : 텍스트 임베딩만 추가한 모델
# MAGIC

# COMMAND ----------

train_df_2.printSchema


# COMMAND ----------

train_df_3.printSchema

# COMMAND ----------

train_df_3 = train_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2','target'])
train_df_3 = train_df_3.na.drop()
test_df_3 =  test_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2','target'])
test_df_3 = test_df_3.na.drop()

vectorAssembler_3 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2' ], outputCol="features")

# 임베딩 변수 컬럼으로 변환하는 것 필요함
# 배열 컬럼 이름 리스트
array_columns = ['new_pcaValues32_1', 'new_pcaValues32_2']

# 새로운 컬럼과 이에 해당하는 원래 컬럼의 매핑 딕셔너리
new_columns = {}

from pyspark.sql.functions import expr

# 배열 컬럼을 분해하여 각 요소를 개별 컬럼으로 만듦
for array_col in array_columns:
    for i in range(32):
        # 새로운 컬럼 이름 생성
        new_col_name = f"{array_col}_{i+1}"
        new_columns[new_col_name] = array_col
        # selectExpr을 사용하여 각 요소를 개별 컬럼으로 생성
        train_df_3 = train_df_3.withColumn(new_col_name, col(array_col)[i].cast("int"))
        test_df_3 = test_df_3.withColumn(new_col_name, col(array_col)[i].cast("int"))

train_df_3 = train_df_3.drop(*array_columns)
test_df_3 = test_df_3.drop(*array_columns)
        
    
# 각 배열 요소를 새로운 컬럼으로 변환
#for new_col_name, array_col in new_columns.items():
#    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
#    i = int(new_col_name.split("_")[-1]) - 1
#    train_df_3 = train_df_3.withColumn(new_col_name, col(array_col)[i])

#for new_col_name, array_col in new_columns.items():
#    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
#    i = int(new_col_name.split("_")[-1]) - 1
#    test_df_3 = test_df_3.withColumn(new_col_name, col(array_col)[i])

#columns_to_drop = ['new_pcaValues32_1', 'new_pcaValues32_2']
#train_df_3 = train_df_3.drop(*columns_to_drop)
#test_df_3 = test_df_3.drop(*columns_to_drop)


from pyspark.ml.feature import VectorAssembler

# 새롭게 생성된 컬럼들을 포함한 전체 컬럼 리스트 생성
assembler_columns = ['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity'] + list(new_columns.keys())

# VectorAssembler 인스턴스 생성
vectorAssembler_3 = VectorAssembler(inputCols=assembler_columns, outputCol="features")

# COMMAND ----------

# 불균형 가중치 
numPositives = train_df_3.filter(train_df_3["target"] == 1).count()
numNegatives = train_df_3.filter(train_df_3["target"] == 0).count()
total = train_df_3.count()

balanceRatio = numNegatives / total

train_df_3 = train_df_3.withColumn('classWeight', when(train_df_3['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_3, gbt])
model_3 = pipeline.fit(train_df_3)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_3 = model_3.transform(test_df_3)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_3)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_3)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_3)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_3)
recall = evaluatorRecall.evaluate(predictions_3)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_3 = "dbfs:/FileStore/amazon/model/model_3"
model_3.write().overwrite().save(model_path_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4번 모델링 : 이미지 유사도만 추가한 모델

# COMMAND ----------

# 이미지 없는 null 값가진 행 제거한 데이터 셋
df_5_cos_null_image = df_5_cos.na.drop(subset=["image_1"])
df_5_cos_null_image = df_5_cos_null_image.na.drop(subset=["image_2"])
df_5_cos_null_image = df_5_cos_null_image.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth',
 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec',
 'price_similarity', 'target', 'image_1', 'image_2', 'image_cosine_similarity','target'])

# train 데이터와 test 데이터 나누기
fractions = df_5_cos_null_image.select("target").distinct().rdd.flatMap(lambda x: x).collect()
fractions = {row: 0.8 for row in fractions}  # 트레인셋 80%

# `sampleBy` 함수를 사용하여 트레인셋 추출
train_df_image = df_5_cos_null_image.sampleBy("target", fractions, seed=42)

# `exceptAll`을 이용해서 트레인셋에 없는 행들을 테스트셋으로 설정
test_df_image = df_5_cos_null_image.exceptAll(train_df_image)

# 결과 확인
print(f"Total count: {df_5_cos_null_image.count()}")
print(f"Train count: {train_df_image.count()}, Test count: {test_df_image.count()}")

# COMMAND ----------

train_df_4 = train_df_image.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity','target'])
train_df_4 = train_df_4.na.drop()
test_df_4 = test_df_image.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity','target'])
test_df_4 = test_df_4.na.drop()

vectorAssembler_4 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity'], outputCol="features")


# 불균형 가중치 
numPositives = train_df_4.filter(train_df_4["target"] == 1).count()
numNegatives = train_df_4.filter(train_df_4["target"] == 0).count()
total = train_df_4.count()

balanceRatio = numNegatives / total

train_df_4 = train_df_4.withColumn('classWeight', when(train_df_4['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_4, gbt])
model_4 = pipeline.fit(train_df_4)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_4 = model_4.transform(test_df_4)

# COMMAND ----------


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_4)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_4)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_4)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_4)
recall = evaluatorRecall.evaluate(predictions_4)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_4 = "dbfs:/FileStore/amazon/model/model_4"
model_4.write().overwrite().save(model_path_4)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 5번 모델링 : 이미지 임베딩만 추가한 모델
# MAGIC

# COMMAND ----------

train_df_5 = train_df_image.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_1', 'image_2','target'])
train_df_5 = train_df_5.na.drop()
test_df_5 =  test_df_image.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_1', 'image_2','target'])
test_df_5 = test_df_5.na.drop()

vectorAssembler_5 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_1', 'image_2'], outputCol="features")

# 임베딩 변수 컬럼으로 변환하는 것 필요함
# 배열 컬럼 이름 리스트
array_columns = ['image_1', 'image_2']

# 새로운 컬럼과 이에 해당하는 원래 컬럼의 매핑 딕셔너리
new_columns = {}

from pyspark.sql.functions import expr

# 배열 컬럼을 분해하여 각 요소를 개별 컬럼으로 만듦
for array_col in array_columns:
    for i in range(1024):
        # 새로운 컬럼 이름 생성
        new_col_name = f"{array_col}_{i+1}"
        new_columns[new_col_name] = array_col
        # selectExpr을 사용하여 각 요소를 개별 컬럼으로 생성
        train_df_5 = train_df_5.withColumn(new_col_name, col(array_col)[i].cast("int"))
        test_df_5 = test_df_5.withColumn(new_col_name, col(array_col)[i].cast("int"))
        
train_df_5 = train_df_5.drop(*array_columns)
test_df_5 = test_df_5.drop(*array_columns)

#for array_col in array_columns:
#    for i in range(1024):
#        # 새 컬럼 이름: 원래 배열 컬럼 이름 + "_" + 원소 순서(index)
#        new_col_name = f"{array_col}_{i+1}"
#        # 새 컬럼 딕셔너리에 추가
#        new_columns[new_col_name] = array_col
    
# 각 배열 요소를 새로운 컬럼으로 변환
#for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
#    i = int(new_col_name.split("_")[-1]) - 1
#    train_df_5 = train_df_5.withColumn(new_col_name, col(array_col)[i])

#for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
#    i = int(new_col_name.split("_")[-1]) - 1
#    test_df_5 = test_df_5.withColumn(new_col_name, col(array_col)[i])

#columns_to_drop = ['image_1', 'image_2']
#train_df_5 = train_df_5.drop(*columns_to_drop)
#test_df_5 = test_df_5.drop(*columns_to_drop)

from pyspark.ml.feature import VectorAssembler

# 새롭게 생성된 컬럼들을 포함한 전체 컬럼 리스트 생성
assembler_columns = ['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity'] + list(new_columns.keys())

# VectorAssembler 인스턴스 생성
vectorAssembler_5 = VectorAssembler(inputCols=assembler_columns, outputCol="features")

# COMMAND ----------

# 불균형 가중치 
numPositives = train_df_5.filter(train_df_5["target"] == 1).count()
numNegatives = train_df_5.filter(train_df_5["target"] == 0).count()
total = train_df_5.count()

balanceRatio = numNegatives / total

train_df_5 = train_df_5.withColumn('classWeight', when(train_df_5['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_5, gbt])
model_5 = pipeline.fit(train_df_5)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_5 = model_5.transform(test_df_5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_5)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_5)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_5)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_5)
recall = evaluatorRecall.evaluate(predictions_5)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_5 = "dbfs:/FileStore/amazon/model/model_5"
model_5.write().overwrite().save(model_path_5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6번 모델링 : 1&4 - 유사도만

# COMMAND ----------

df_total = df_5_cos.select([
    'cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'target', 'new_pcaValues32_1', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'new_pcaValues32_2', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity', 'cosine_fin'])
df_total = df_total.na.drop(subset=["image_1","image_2","new_pcaValues32_1","new_pcaValues32_2"])

# train 데이터와 test 데이터 나누기
fractions = df_total.select("target").distinct().rdd.flatMap(lambda x: x).collect()
fractions = {row: 0.8 for row in fractions}  # 트레인셋 80%

# `sampleBy` 함수를 사용하여 트레인셋 추출
train_df_to = df_total.sampleBy("target", fractions, seed=42)

# `exceptAll`을 이용해서 트레인셋에 없는 행들을 테스트셋으로 설정
test_df_to = df_total.exceptAll(train_df_to)

# 결과 확인
print(f"Total count: {df_total.count()}")
print(f"Train count: {train_df_to.count()}, Test count: {test_df_to.count()}")

# COMMAND ----------

train_df_6 = train_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
train_df_6 = train_df_6.na.drop()
test_df_6 = test_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
test_df_6 = test_df_6.na.drop()

vectorAssembler_6 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity',
 'kurtosis_cosine_similarity', 'image_cosine_similarity'], outputCol="features")


# 불균형 가중치 
numPositives = train_df_6.filter(train_df_6["target"] == 1).count()
numNegatives = train_df_6.filter(train_df_6["target"] == 0).count()
total = train_df_6.count()

balanceRatio = numNegatives / total

train_df_6 = train_df_6.withColumn('classWeight', when(train_df_6['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_6, gbt])
model_6 = pipeline.fit(train_df_6)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_6 = model_6.transform(test_df_6)

# COMMAND ----------


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_6)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_6)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_6)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_6)
recall = evaluatorRecall.evaluate(predictions_6)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_6 = "dbfs:/FileStore/amazon/model/model_6"
model_6.write().overwrite().save(model_path_6)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7번 모델링 : 2&5 - 통계량 임베딩- 임베딩만  (이건 컬럼이 진짜 너무너무 많아서 제외)
# MAGIC

# COMMAND ----------

train_df_7 = train_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2','target' ])
train_df_7 = train_df_7.na.drop()
test_df_7 = test_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2','target' ])
test_df_7 = test_df_7.na.drop()

vectorAssembler_7 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2' ], outputCol="features")


# 임베딩 변수 컬럼으로 변환하는 것 필요함
# 배열 컬럼 이름 리스트
array_columns = [ 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2']

# 새로운 컬럼과 이에 해당하는 원래 컬럼의 매핑 딕셔너리
new_columns = {}

for array_col in array_columns:
    # 32개 원소에 대해 각각 반복
    for i in range(32):
        # 새 컬럼 이름: 원래 배열 컬럼 이름 + "_" + 원소 순서(index)
        new_col_name = f"{array_col}_{i+1}"
        # 새 컬럼 딕셔너리에 추가
        new_columns[new_col_name] = array_col
    
# 각 배열 요소를 새로운 컬럼으로 변환
for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
    i = int(new_col_name.split("_")[-1]) - 1
    train_df_7 = train_df_7.withColumn(new_col_name, col(array_col)[i])

for new_col_name, array_col in new_columns.items():
    # 배열의 인덱스 i를 사용하여 새 컬럼 생성
    i = int(new_col_name.split("_")[-1]) - 1
    test_df_7 = test_df_7.withColumn(new_col_name, col(array_col)[i])

columns_to_drop = ['image_1', 'image_2']
train_df_7 = train_df_7.drop(*columns_to_drop)
test_df_7 = test_df_7.drop(*columns_to_drop)

from pyspark.ml.feature import VectorAssembler

# 새롭게 생성된 컬럼들을 포함한 전체 컬럼 리스트 생성
assembler_columns = ['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity'] + list(new_columns.keys())

# VectorAssembler 인스턴스 생성
vectorAssembler_7 = VectorAssembler(inputCols=assembler_columns, outputCol="features")

# 불균형 가중치 
numPositives = train_df_7.filter(train_df_7["target"] == 1).count()
numNegatives = train_df_7.filter(train_df_7["target"] == 0).count()
total = train_df_7.count()

balanceRatio = numNegatives / total

train_df_7 = train_df_7.withColumn('classWeight', when(train_df_7['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_7, gbt])
model_7 = pipeline.fit(train_df_7)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_7 = model_7.transform(test_df_7)

# COMMAND ----------


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_7)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_7)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_7)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_7)
recall = evaluatorRecall.evaluate(predictions_7)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 8번 모델링 : 3&5 - 텍스트 임베딩 - 이미지 임베딩
# MAGIC

# COMMAND ----------

train_df_8 = train_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2','target'])
train_df_8 = train_df_8.na.drop()
test_df_8 = test_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2','target'])
test_df_8 = test_df_8.na.drop()

vectorAssembler_8 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2'], outputCol="features")


# 임베딩 변수 컬럼으로 변환하는 것 필요함
# 배열 컬럼 이름 리스트
array_columns = ['new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2']

from pyspark.sql.functions import col

# 배열 컬럼 이름과 해당 원소 개수의 매핑
array_columns = {
    'new_pcaValues32_1': 32, 
    'new_pcaValues32_2': 32, 
    'image_1': 1024, 
    'image_2': 1024
}

new_columns = {}

for array_col, num_elements in array_columns.items():
    for i in range(num_elements):
        new_col_name = f"{array_col}_{i+1}"
        new_columns[new_col_name] = array_col

# 각 배열 요소를 새로운 컬럼으로 변환 (train_df_8에 대하여)
for new_col_name, array_col in new_columns.items():
    i = int(new_col_name.split("_")[-1]) - 1
    train_df_8 = train_df_8.withColumn(new_col_name, col(array_col)[i].cast("int"))
    test_df_8 = test_df_8.withColumn(new_col_name, col(array_col)[i].cast("int"))


# 더 이상 필요하지 않은 원본 배열 컬럼 제거
columns_to_drop = ['new_pcaValues32_1', 'new_pcaValues32_2', 'image_1', 'image_2']
train_df_8 = train_df_8.drop(*columns_to_drop)
test_df_8 = test_df_8.drop(*columns_to_drop)

from pyspark.ml.feature import VectorAssembler

# 새롭게 생성된 컬럼들을 포함한 전체 컬럼 리스트 생성
assembler_columns = ['cat2_same','cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity'] + list(new_columns.keys())

# VectorAssembler 인스턴스 생성
vectorAssembler_8 = VectorAssembler(inputCols=assembler_columns, outputCol="features")

# 불균형 가중치 
numPositives = train_df_8.filter(train_df_8["target"] == 1).count()
numNegatives = train_df_8.filter(train_df_8["target"] == 0).count()
total = train_df_8.count()

balanceRatio = numNegatives / total

train_df_8 = train_df_8.withColumn('classWeight', when(train_df_8['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_8, gbt])
model_8 = pipeline.fit(train_df_8)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_8 = model_8.transform(test_df_8)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_8)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_8)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_8)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_8)
recall = evaluatorRecall.evaluate(predictions_8)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

model_path_8 = "dbfs:/FileStore/amazon/model/model_8"
model_8.write().overwrite().save(model_path_8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9번 모델링 : 텍스트 평균 유사도 & 이미지 유사도

# COMMAND ----------

train_df_9 = train_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity', 'cosine_fin','target' ])
train_df_9 = train_df_9.na.drop()
test_df_9 = test_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity', 'cosine_fin','target' ])
test_df_9 = test_df_9.na.drop()

vectorAssembler_9 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'image_cosine_similarity', 'cosine_fin' ], outputCol="features")

# 불균형 가중치 
numPositives = train_df_9.filter(train_df_9["target"] == 1).count()
numNegatives = train_df_9.filter(train_df_9["target"] == 0).count()
total = train_df_9.count()

balanceRatio = numNegatives / total

train_df_9 = train_df_9.withColumn('classWeight', when(train_df_9['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_9, gbt])
model_9 = pipeline.fit(train_df_9)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_9 = model_9.transform(test_df_9)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_9)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_9)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_9)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_9)
recall = evaluatorRecall.evaluate(predictions_9)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

#model_path_7 = "path/to/save/model_7"
#model_7.write().overwrite().save(model_path)

model_path_9 = "dbfs:/FileStore/amazon/model/model_9"
model_9.write().overwrite().save(model_path_9)

# COMMAND ----------

from pyspark.ml import PipelineModel

# 저장된 모델 불러오기
loaded_model_1_we = PipelineModel.load(model_path_1_we)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 최종 선택 모델 및 변수
# MAGIC -
# MAGIC
# MAGIC #### 파라미터 튜닝 진행
# MAGIC -
# MAGIC
# MAGIC #### 시간 되면 다른 모델도 진행
# MAGIC -
# MAGIC
# MAGIC #### 모델 저장
# MAGIC -
# MAGIC
# MAGIC #### 원래 5백만개  train  데이터에 확정된 변수를 옆으로 합친 후에, 해당 모델로 prediction 진행
# MAGIC -
# MAGIC
# MAGIC #### 그 값으로 랭킹평가 진행
# MAGIC -
# MAGIC

# COMMAND ----------



# COMMAND ----------

# 원래의 5백 몇만개 train 셋 불러오기
real_train = spark.read.table("asac.240430_train_df")

# 랭킹평가 위한 test 셋 불러오기
real_test = spark.read.table("asac.240430_test_df")

# COMMAND ----------


