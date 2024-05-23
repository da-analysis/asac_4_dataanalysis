# Databricks notebook source
# MAGIC %md
# MAGIC ### 1만 * 1만개의 asin 조합과 텍스트 리뷰 유사도 확인하기

# COMMAND ----------

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
from pyspark.sql.functions import desc

# COMMAND ----------

df = spark.read.table("asac.embed_cell_sbert_32_fin")

# COMMAND ----------

df.columns

# COMMAND ----------

df = df.drop("new_pcaValues32")

# COMMAND ----------

df_10000 = df.orderBy(desc("list_length")).limit(10000)

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import desc

df_10000_asin1 = df_10000.withColumnRenamed("asin", "asin1")
df_10000_asin2 = df_10000.withColumnRenamed("asin", "asin2")
for column_name in df_10000_asin1.columns:
    df_10000_asin1 = df_10000_asin1.withColumnRenamed(column_name, column_name + "_1")

# df_10000_asin2의 모든 컬럼 이름에 _2를 추가
for column_name in df_10000_asin2.columns:
    df_10000_asin2 = df_10000_asin2.withColumnRenamed(column_name, column_name + "_2")

# 이제 컬럼 이름이 변경된 두 데이터프레임을 crossJoin 수행
df_cross_joined = df_10000_asin1.crossJoin(df_10000_asin2)
df_cross_joined = df_cross_joined.withColumnRenamed("asin1_1", "asin1")
df_cross_joined = df_cross_joined.withColumnRenamed("asin2_2", "asin2")

df_combinations = df_cross_joined.filter(col("asin1") != col("asin2"))

# COMMAND ----------

df_combinations.columns

# COMMAND ----------

name = "asac.text_sim_1"
df_combinations.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text_sim_1
# MAGIC limit 10

# COMMAND ----------

df_feat = spark.read.table("asac.text_sim_1")

# COMMAND ----------

df_feat.columns

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 코사인 유사도 구하기
# MAGIC

# COMMAND ----------

## 방법1
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
    df_feat = df_feat.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    df_feat = df_feat.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
df_feat = df_feat.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

name = "asac.text_sim_2"
df_feat.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 리뷰 텍스트 붙이기

# COMMAND ----------

df_feat = spark.read.table("asac.text_sim_2")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text_sim_2
# MAGIC limit 10

# COMMAND ----------

df = spark.read.table("asac.review_cellphone_accessories_final")

# COMMAND ----------

col = ["mean_vector_1", "mean_vector_2","std_dev_1", "std_dev_2","q1_1", "q1_2",
    "q2_1", "q2_2","q3_1", "q3_2","skewness_1", "skewness_2","kurtosis_1", "kurtosis_2","variance_1","variance_2"]
for c in col:
    df_feat = df_feat.drop(c)

# COMMAND ----------

name = "asac.text_sim_3"
df_feat.write.saveAsTable(name)

# COMMAND ----------

df_feat = spark.read.table("asac.text_sim_3")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text_sim_3
# MAGIC limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.text_sim_3
# MAGIC limit 10    -- 1만 * 1만에서 자기자신 제외

# COMMAND ----------

df_feat.columns

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.review_cellphone_accessories_final
# MAGIC limit 10

# COMMAND ----------

from pyspark.sql.functions import col, collect_list


# df DataFrame에서 df_feat_combined에 있는 asin과 일치하는 행만 필터링
df_filtered = df.join(df_feat, df.asin == df_feat.asin1, "inner")

# asin 컬럼별로 reviewText 값 합치기
df_aggregated = df_filtered.groupBy("asin").agg(collect_list("reviewText").alias("reviewTexts"))

# COMMAND ----------

df_aggregated.columns

# COMMAND ----------

name = "asac.text_sim_2_1"
df_aggregated.write.saveAsTable(name)

# COMMAND ----------

df_aggregated = spark.read.table("asac.text_sim_2_1")

# COMMAND ----------

df_joined = df_feat.join(df_aggregated, df_feat.asin1 == df_aggregated.asin, "left")

# 조인된 데이터프레임에서 reviewText 칼럼의 이름을 reviewText_1로 변경
df_result = df_joined.withColumnRenamed("reviewTexts", "reviewText_1")

# 필요하지 않은 중복 칼럼 제거 (예: df_aggregated의 asin 칼럼)
df_result = df_result.drop(df_aggregated.asin)

# COMMAND ----------

df_result.columns

# COMMAND ----------

name = "asac.text_sim_3"
df_result.write.saveAsTable(name)

# COMMAND ----------

df_result = spark.read.table("asac.text_sim_3")

# COMMAND ----------

df_result = df_result.join(df_aggregated, df_feat.asin2 == df_aggregated.asin, "left")

df_result = df_result.withColumnRenamed("reviewTexts", "reviewText_2")

df_result = df_result.drop(df_aggregated.asin)

# COMMAND ----------

df_result.columns

# COMMAND ----------

df_result = df_result.drop("asin")

# COMMAND ----------

df_result.show(1)

# COMMAND ----------

name = "asac.text_sim_fin"
df_result.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text_sim_fin
# MAGIC limit 10

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 5천 * 5천 으로 다시해보기

# COMMAND ----------

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
from pyspark.sql.functions import desc

# COMMAND ----------

df = spark.read.table("asac.embed_cell_sbert_32_fin")
df.columns

# COMMAND ----------

df = df.drop("new_pcaValues32")
df_5000 = df.orderBy(desc("list_length")).limit(5000)

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import desc

df_5000_asin1 = df_5000.withColumnRenamed("asin", "asin1")
df_5000_asin2 = df_5000.withColumnRenamed("asin", "asin2")
for column_name in df_5000_asin1.columns:
    df_5000_asin1 = df_5000_asin1.withColumnRenamed(column_name, column_name + "_1")

# df_5000_asin2의 모든 컬럼 이름에 _2를 추가
for column_name in df_5000_asin2.columns:
    df_5000_asin2 = df_5000_asin2.withColumnRenamed(column_name, column_name + "_2")

# 이제 컬럼 이름이 변경된 두 데이터프레임을 crossJoin 수행
df_cross_joined = df_5000_asin1.crossJoin(df_5000_asin2)
df_cross_joined = df_cross_joined.withColumnRenamed("asin1_1", "asin1")
df_cross_joined = df_cross_joined.withColumnRenamed("asin2_2", "asin2")

df_combinations = df_cross_joined.filter(col("asin1") != col("asin2"))

# COMMAND ----------

df_combinations.columns

# COMMAND ----------

name = "asac.text5000_sim_1"
df_combinations.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text5000_sim_1
# MAGIC limit 10

# COMMAND ----------

df_feat = spark.read.table("asac.text5000_sim_1")

# COMMAND ----------

df_feat.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### 코사인

# COMMAND ----------

## 방법1
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
    df_feat = df_feat.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    df_feat = df_feat.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
df_feat = df_feat.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

col = ["mean_vector_1", "mean_vector_2","std_dev_1", "std_dev_2","q1_1", "q1_2",
    "q2_1", "q2_2","q3_1", "q3_2","skewness_1", "skewness_2","kurtosis_1", "kurtosis_2","variance_1","variance_2"]
for c in col:
    df_feat = df_feat.drop(c)

# COMMAND ----------

name = "asac.text5000_sim_2"
df_feat.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 텍스트

# COMMAND ----------

df_feat = spark.read.table("asac.text5000_sim_2")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text5000_sim_2
# MAGIC limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac5000.text_sim_2

# COMMAND ----------

df = spark.read.table("asac.review_cellphone_accessories_final")

# COMMAND ----------

from pyspark.sql.functions import col, collect_list


# df DataFrame에서 df_feat_combined에 있는 asin과 일치하는 행만 필터링
df_filtered = df.join(df_feat, df.asin == df_feat.asin1, "inner")

# asin 컬럼별로 reviewText 값 합치기
df_aggregated = df_filtered.groupBy("asin").agg(collect_list("reviewText").alias("reviewTexts"))

# COMMAND ----------

name = "asac.text5000_sim_2_1"
df_aggregated.write.szhaveAsTable(name)

# COMMAND ----------

df_aggregated = spark.read.table("asac.text5000_sim_2_1")

# COMMAND ----------

df_joined = df_feat.join(df_aggregated, df_feat.asin1 == df_aggregated.asin, "left")

# 조인된 데이터프레임에서 reviewText 칼럼의 이름을 reviewText_1로 변경
df_result = df_joined.withColumnRenamed("reviewTexts", "reviewText_1")

# 필요하지 않은 중복 칼럼 제거 (예: df_aggregated의 asin 칼럼)
df_result = df_result.drop(df_aggregated.asin)

# COMMAND ----------

df_result.columns

# COMMAND ----------

name = "asac.text5000_sim_3"
df_result.write.saveAsTable(name)

# COMMAND ----------

df_result = spark.read.table("asac.text5000_sim_3")

# COMMAND ----------

df_result = df_result.join(df_aggregated, df_feat.asin2 == df_aggregated.asin, "left")

df_result = df_result.withColumnRenamed("reviewTexts", "reviewText_2")

df_result = df_result.drop(df_aggregated.asin)

# COMMAND ----------

df_result.columns

# COMMAND ----------

df_result = df_result.drop("asin")

# COMMAND ----------

name = "asac.text5000_sim_fin"
df_result.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text5000_sim_fin
# MAGIC limit 10

# COMMAND ----------


