# Databricks notebook source
# MAGIC %md
# MAGIC ### 이미지 임베딩 델타 테이블 저장 (asin, 임베딩)

# COMMAND ----------

# MAGIC %pip install pandas

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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# 텍스트
df_text = spark.read.table("asac.embed_cell_sbert_32_fin")

# 5배수
df_5 = spark.read.table("asac.240430_review3_5multiply")

## 완전 전체 train 셋과 test 셋
total_train = spark.read.table("asac.240430_train_df")
total_test =  spark.read.table("asac.240430_test_df")
df_5_cos = spark.read.table("asac.df_5_cos")

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

name = "asac.image_emb"
df_image.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %md
# MAGIC - asin으로 zorder 걸기

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.image_emb

# COMMAND ----------

# MAGIC %sql
# MAGIC CONVERT TO DELTA hive_metastore.asac.image_emb;
# MAGIC OPTIMIZE hive_metastore.asac.image_emb
# MAGIC ZORDER BY asin;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 텍스트, 트레인 zorder 걸고나서 저장

# COMMAND ----------

# MAGIC %sql
# MAGIC CONVERT TO DELTA hive_metastore.asac.embed_cell_sbert_32_fin;
# MAGIC OPTIMIZE hive_metastore.asac.embed_cell_sbert_32_fin
# MAGIC ZORDER BY asin;

# COMMAND ----------

# MAGIC %sql
# MAGIC CONVERT TO DELTA hive_metastore.asac.240430_train_df;
# MAGIC OPTIMIZE hive_metastore.asac.240430_train_df
# MAGIC ZORDER BY (asin1,asin2);

# COMMAND ----------

# 텍스트 임베딩, 통계량 임베딩 
df_text = spark.read.table("asac.embed_cell_sbert_32_fin")

# 이미지 임베딩
df_image = spark.read.table("asac.image_emb")

## 완전 전체 train 셋
total_train = spark.read.table("asac.240430_train_df")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 텍스트 칼럼 추가

# COMMAND ----------

total_train_embedding = spark.sql("""
          select /*+ BROADCAST(b) */ 
                t.*
                    , c.list_length as list_length_2
                    , c.mean_vector as mean_vector_2
                    , c.std_dev as std_dev_2
                    , c.q1 as q1_2
                    , c.q2 as q2_2
                    , c.q3 as q3_2
                    , c.skewness as skewness_2
                    , c.kurtosis as kurtosis_2    
          from(
          select a.*
                    , b.list_length as list_length_1
                    , b.mean_vector as mean_vector_1
                    , b.std_dev as std_dev_1
                    , b.q1 as q1_1
                    , b.q2 as q2_1
                    , b.q3 as q3_1
                    , b.skewness as skewness_1
                    , b.kurtosis as kurtosis_1    
          from asac.240430_train_df as a
          left join asac.embed_cell_sbert_32_fin as b
          on 1=1
          and a.asin1 = b.asin
          ) as t
          left join asac.embed_cell_sbert_32_fin as c
          on 1=1
          and t.asin2 = c.asin
""")

# COMMAND ----------

name = "asac.total_train_temp1"
total_train_embedding.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.total_train_temp1

# COMMAND ----------

# # asin1과 asin2 를 기준으로 left outer join 진행
# # 임베딩, 통계량 값, 길이

# col_names = total_train.columns

# total_train = total_train.join(df_text, total_train.asin1 == df_text.asin,"left_outer")
# total_train = total_train.drop("asin")

# for col_name in total_train.columns:
#     if col_name not in col_names:  
#         total_train = total_train.withColumnRenamed(col_name, col_name + "_1")

# col_names = total_train.columns

# total_train = total_train.join(df_text, total_train.asin2 == df_text.asin,"left_outer")
# total_train = total_train.drop("asin")

# for col_name in total_train.columns:
#     if col_name not in col_names:  
#         total_train = total_train.withColumnRenamed(col_name, col_name + "_2")

# total_train = total_train.drop("asin")
# total_train = total_train.drop("variance_1")
# total_train = total_train.drop("variance_2")

# COMMAND ----------

# name = "asac.total_train_temp1"
# total_train.write.saveAsTable(name)

# COMMAND ----------

total_train = spark.read.table("asac.total_train_temp1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 이미지 컬럼 추가

# COMMAND ----------

from pyspark.sql.functions import col

total_train = total_train.join(df_image.alias("image1"), total_train.asin1 == col("image1.asin"), "left_outer")
total_train = total_train.withColumnRenamed("embedding_array", "image_1")
total_train = total_train.drop("asin")

total_train = total_train.join(df_image.alias("image2"), total_train.asin2 == col("image2.asin"), "left_outer")
total_train = total_train.withColumnRenamed("embedding_array", "image_2")
total_train = total_train.drop("asin")

# COMMAND ----------

name = "asac.total_train_temp2"
total_train.write.saveAsTable(name)

# COMMAND ----------

total_train_temp = spark.read.table("asac.total_train_temp2")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.total_train_temp2

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp2

# COMMAND ----------

# MAGIC %md
# MAGIC ### 코사인 유사도 계산

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
    total_train_temp = total_train_temp.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    total_train_temp = total_train_temp.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
total_train_temp = total_train_temp.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

name = "asac.total_train_temp3"
total_train_temp.write.saveAsTable(name)

# COMMAND ----------

total_train_temp_te = spark.read.table("asac.total_train_temp3")

# COMMAND ----------

# 방법2
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
from pyspark.sql.functions import col, when

for col1, col2 in columns:
    total_train_temp = total_train_temp.withColumn(
        f"{col1[:-2]}_cosine_similarity",
        when(
            col(col1).isNull() | col(col2).isNull(),
            0
        ).otherwise(cosine_similarity_udf(col(col1), col(col2)))
    )
    total_train_temp = total_train_temp.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# 최종 코사인 유사도 평균 계산
total_train_temp = total_train_temp.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------



# COMMAND ----------

## 방법1
from pyspark.sql.functions import expr, col
from pyspark.sql.functions import col, sqrt, sum as _sum, when
columns = [
    ("image_1", "image_2")
]

# 각 컬럼 쌍에 대해 반복
for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(1000)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(1000)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(1000)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # DataFrame에 코사인 유사도 컬럼 추가
    total_train_temp_image = total_train_temp_te.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    total_train_temp_image = total_train_temp_image.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# COMMAND ----------

# 방법2
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, col, when, PandasUDFType
from pyspark.sql.types import DoubleType


columns = [ ("image_1", "image_2")
]

@pandas_udf("double", PandasUDFType.SCALAR)
def cosine_similarity_udf(v1: pd.Series, v2: pd.Series) -> pd.Series:
    # 결과를 저장할 빈 리스트 초기화
    cosine_similarity_results = []
    
    for a, b in zip(v1, v2):
        # a 또는 b가 None이면 코사인 유사도를 계산하지 않고 0을 추가
        if a is None or b is None:
            cosine_similarity_results.append(0)
        else:
            # 벡터 a와 b 사이의 코사인 유사도 계산
            dot_product = np.dot(a, b)
            norm_a = np.sqrt(np.dot(a, a))
            norm_b = np.sqrt(np.dot(b, b))
            cosine_similarity = dot_product / (norm_a * norm_b)
            cosine_similarity_results.append(cosine_similarity)
    
    return pd.Series(cosine_similarity_results)


# DataFrame에 코사인 유사도 컬럼 추가
from pyspark.sql.functions import col, when

for col1, col2 in columns:
    total_train_temp_image = total_train_temp_te.withColumn(
        f"{col1[:-2]}_cosine_similarity",
        when(
            col(col1).isNull() | col(col2).isNull(),
            0
        ).otherwise(cosine_similarity_udf(col(col1), col(col2)))
    )
    total_train_temp_image = total_train_temp_image.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# COMMAND ----------

name = "asac.total_train_temp_fin"
total_train_temp_image.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.total_train_temp_fin

# COMMAND ----------

total_train_df = spark.read.table("asac.total_train_temp_fin")

# COMMAND ----------

# 텍스트 없는 행 삭제
filtered_df_text = total_train_df.filter(total_train_df.mean_vector_cosine_similarity!= 0)
name = "asac.total_train_temp_fin_text"
filtered_df_text.write.saveAsTable(name)

# COMMAND ----------

# 이미지 없는 행 삭제
filtered_df_image = total_train_df.filter(total_train_df.image_cosine_similarity  != 0)
name = "asac.total_train_temp_fin_image"
filtered_df_image.write.saveAsTable(name)

# COMMAND ----------

# 이미지랑 텍스트 없는 행 삭제
filtered_df = filtered_df_text.filter(filtered_df_text.image_cosine_similarity != 0)
name = "asac.total_train_temp_fin_text_image"
filtered_df.write.saveAsTable(name)

# COMMAND ----------

text = spark.read.table("asac.total_train_temp_fin_text")
image = spark.read.table("asac.total_train_temp_fin_image")
comb = spark.read.table("asac.total_train_temp_fin_text_image")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin_text  --322

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin_image  --1,289,966

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin_text_image  -- 1,290,158

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin_text_image
# MAGIC where mean_vector_cosine_similarity == 0

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.total_train_temp_fin_text_image
# MAGIC where image_cosine_similarity == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-1번 모델로 예측 진행

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# 데이터 준비 및 전처리
vectorAssembler_1 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity'], outputCol="features")

# COMMAND ----------

train_df_1 = text.select(['asin1','asin2','review_cnts','cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()

from pyspark.ml import PipelineModel

# 모델 저장 경로
model_path_1_we = "dbfs:/FileStore/amazon/model/model_1_we"

# 저장된 모델 불러오기
loaded_model_1_we = PipelineModel.load(model_path_1_we)

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

predictions_prob_1 = loaded_model_1_we.transform(train_df_1)
extract_prob_udf = udf(extract_prob, DoubleType())

predictions_prob_1 = predictions_prob_1.withColumn("prob", extract_prob_udf(col("probability")))


# COMMAND ----------

import numpy as np
prob_list = predictions_prob_1.select("prob").rdd.flatMap(lambda x: x).collect()

# 리스트를 numpy 배열로 변환
prob_array = np.array(prob_list)

# matplotlib를 이용한 시각화
plt.figure(figsize=(10, 6))
plt.hist(prob_array, bins=20, density=True, alpha=0.6, color='b')
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('density') #freq 아니고 density
plt.show()

# COMMAND ----------

selected_df_1 = predictions_prob_1.select('asin1', 'asin2', 'prob','review_cnts')
name = "asac.pred_1_we"
selected_df_1.write.mode('overwrite').saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.pred_1_we

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 6번 모델로 예측 진행

# COMMAND ----------

train_df_6 = comb.select(['asin1','asin2','review_cnts','cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
train_df_6 = train_df_6.na.drop()

# COMMAND ----------

from pyspark.ml import PipelineModel

# 모델 저장 경로
model_path_6 = "dbfs:/FileStore/amazon/model/model_6"

# 저장된 모델 불러오기
loaded_model_6 = PipelineModel.load(model_path_6)

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

predictions_prob_6= loaded_model_6.transform(train_df_6)
extract_prob_udf = udf(extract_prob, DoubleType())

predictions_prob_6 = predictions_prob_6.withColumn("prob", extract_prob_udf(col("probability")))

# COMMAND ----------

import numpy as np
prob_list = predictions_prob_6.select("prob").rdd.flatMap(lambda x: x).collect()

# 리스트를 numpy 배열로 변환
prob_array = np.array(prob_list)

# matplotlib를 이용한 시각화
plt.figure(figsize=(10, 6))
plt.hist(prob_array, bins=20, density=True, alpha=0.6, color='b')
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('density') #dendity임
plt.show()

# COMMAND ----------

selected_df_6 = predictions_prob_6.select('asin1', 'asin2', 'prob','review_cnts')
name = "asac.pred_6"
selected_df_6.write.mode('overwrite').saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.pred_6

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-1번 모델 랭킹평가 진행

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath('/Workspace/Shared/function/eval.py'))

# COMMAND ----------

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np

try:
    from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
    from pyspark.sql import Window, DataFrame
    from pyspark.sql.functions import col, row_number, expr
    from pyspark.sql.functions import udf
    import pyspark.sql.functions as F
    from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField
    from pyspark.ml.linalg import VectorUDT
except ImportError:
    pass  # skip this import if we are in pure python environment

from recommenders.utils.constants import (
    DEFAULT_PREDICTION_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_RELEVANCE_COL,
    DEFAULT_SIMILARITY_COL,
    DEFAULT_ITEM_FEATURES_COL,
    DEFAULT_ITEM_SIM_MEASURE,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)


class SparkRatingEvaluation:
    """Spark Rating Evaluator"""

    def __init__(
        self,
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    ):
        """Initializer.

        This is the Spark version of rating metrics evaluator.
        The methods of this class, calculate rating metrics such as root mean squared error, mean absolute error,
        R squared, and explained variance.

        Args:
            rating_true (pyspark.sql.DataFrame): True labels.
            rating_pred (pyspark.sql.DataFrame): Predicted labels.
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        # Check if inputs are Spark DataFrames.
        if not isinstance(self.rating_true, DataFrame):
            raise TypeError(
                "rating_true should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        if not isinstance(self.rating_pred, DataFrame):
            raise TypeError(
                "rating_pred should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        # Check if columns exist.
        true_columns = self.rating_true.columns
        pred_columns = self.rating_pred.columns

        if rating_true.count() == 0:
            raise ValueError("Empty input dataframe")
        if rating_pred.count() == 0:
            raise ValueError("Empty input dataframe")

        if self.col_user not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing User Col")
        if self.col_item not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Item Col")
        if self.col_rating not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Rating Col")

        if self.col_user not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing User Col"
            )  # pragma : No Cover
        if self.col_item not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing Item Col"
            )  # pragma : No Cover
        if self.col_prediction not in pred_columns:
            raise ValueError("Schema of rating_pred not valid. Missing Prediction Col")

        self.rating_true = self.rating_true.select(
            col(self.col_user),
            col(self.col_item),
            col(self.col_rating).cast("double").alias("label"),
        )
        self.rating_pred = self.rating_pred.select(
            col(self.col_user),
            col(self.col_item),
            col(self.col_prediction).cast("double").alias("prediction"),
        )

        self.y_pred_true = (
            self.rating_true.join(
                self.rating_pred, [self.col_user, self.col_item], "inner"
            )
            .drop(self.col_user)
            .drop(self.col_item)
        )

        self.metrics = RegressionMetrics(
            self.y_pred_true.rdd.map(lambda x: (x.prediction, x.label))
        )

    def rmse(self):
        """Calculate Root Mean Squared Error.

        Returns:
            float: Root mean squared error.
        """
        return self.metrics.rootMeanSquaredError

    def mae(self):
        """Calculate Mean Absolute Error.

        Returns:
            float: Mean Absolute Error.
        """
        return self.metrics.meanAbsoluteError

    def rsquared(self):
        """Calculate R squared.

        Returns:
            float: R squared.
        """
        return self.metrics.r2

    def exp_var(self):
        """Calculate explained variance.

        Note:
           Spark MLLib's implementation is buggy (can lead to values > 1), hence we use var().

        Returns:
            float: Explained variance (min=0, max=1).
        """
        var1 = self.y_pred_true.selectExpr("variance(label-prediction)").collect()[0][0]
        var2 = self.y_pred_true.selectExpr("variance(label)").collect()[0][0]

        if var1 is None or var2 is None:
            return -np.inf
        else:
            # numpy divide is more tolerant to var2 being zero
            return 1 - np.divide(var1, var2)


class SparkRankingEvaluation:
    """Spark Ranking Evaluator"""

    def __init__(
        self,
        rating_true,
        rating_pred,
        k=DEFAULT_K,
        relevancy_method="top_k",
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        threshold=DEFAULT_THRESHOLD,
    ):
        """Initialization.
        This is the Spark version of ranking metrics evaluator.
        The methods of this class, calculate ranking metrics such as precision@k, recall@k, ndcg@k, and mean average
        precision.

        The implementations of precision@k, ndcg@k, and mean average precision are referenced from Spark MLlib, which
        can be found at `the link <https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems>`_.

        Args:
            rating_true (pyspark.sql.DataFrame): DataFrame of true rating data (in the
                format of customerID-itemID-rating tuple).
            rating_pred (pyspark.sql.DataFrame): DataFrame of predicted rating data (in
                the format of customerID-itemID-rating tuple).
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
            k (int): number of items to recommend to each user.
            relevancy_method (str): method for determining relevant items. Possible
                values are "top_k", "by_time_stamp", and "by_threshold".
            threshold (float): threshold for determining the relevant recommended items.
                This is used for the case that predicted ratings follow a known
                distribution. NOTE: this option is only activated if `relevancy_method` is
                set to "by_threshold".
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.threshold = threshold

        # Check if inputs are Spark DataFrames.
        if not isinstance(self.rating_true, DataFrame):
            raise TypeError(
                "rating_true should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        if not isinstance(self.rating_pred, DataFrame):
            raise TypeError(
                "rating_pred should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        # Check if columns exist.
        true_columns = self.rating_true.columns
        pred_columns = self.rating_pred.columns

        if self.col_user not in true_columns:
            raise ValueError(
                "Schema of rating_true not valid. Missing User Col: "
                + str(true_columns)
            )
        if self.col_item not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Item Col")
        if self.col_rating not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Rating Col")

        if self.col_user not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing User Col"
            )  # pragma : No Cover
        if self.col_item not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing Item Col"
            )  # pragma : No Cover
        if self.col_prediction not in pred_columns:
            raise ValueError("Schema of rating_pred not valid. Missing Prediction Col")

        self.k = k

        relevant_func = {
            "top_k": _get_top_k_items,
            "by_time_stamp": _get_relevant_items_by_timestamp,
            "by_threshold": _get_relevant_items_by_threshold,
        }

        if relevancy_method not in relevant_func:
            raise ValueError(
                "relevancy_method should be one of {}".format(
                    list(relevant_func.keys())
                )
            )

        self.rating_pred = (
            relevant_func[relevancy_method](
                dataframe=self.rating_pred,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_prediction,
                threshold=self.threshold,
            )
            if relevancy_method == "by_threshold"
            else relevant_func[relevancy_method](
                dataframe=self.rating_pred,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_prediction,
                k=self.k,
            )
        )

        self._metrics = self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate ranking metrics."""
        self._items_for_user_pred = self.rating_pred

        self._items_for_user_true = (
            self.rating_true.groupBy(self.col_user)
            .agg(expr("collect_list(" + self.col_item + ") as ground_truth"))
            .select(self.col_user, "ground_truth")
        )

        self._items_for_user_all = self._items_for_user_pred.join(
            self._items_for_user_true, on=self.col_user
        ).drop(self.col_user)

        return RankingMetrics(self._items_for_user_all.rdd)

    def precision_at_k(self):
        """Get precision@k.

        Note:
            More details can be found
            `on the precisionAt PySpark documentation <http://spark.apache.org/docs/3.0.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt>`_.

        Return:
            float: precision at k (min=0, max=1)
        """
        return self._metrics.precisionAt(self.k)

    def recall_at_k(self):
        """Get recall@K.

        NOTE: 
            More details can be found `here <http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.meanAveragePrecision>`_.

        Return:
            float: recall at k (min=0, max=1).
        """
        
        self._items_for_user_all_r = self._items_for_user_true.join(
            self._items_for_user_pred, on=(self._items_for_user_true.asin1 == self._items_for_user_pred.asin1), how='left'
        ).select(F.col('prediction'), F.col('ground_truth'))        

        
        recall = self._items_for_user_all_r.rdd.map(
            lambda x: 0.0 if x[0] is None or x[1] is None else float(len(set(x[0]).intersection(set(x[1])))) / float(len(x[1]))
        ).mean()
        
        return recall

    def ndcg_at_k(self):
        """Get Normalized Discounted Cumulative Gain (NDCG)

        Note:
            More details can be found
            `on the ndcgAt PySpark documentation <http://spark.apache.org/docs/3.0.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.ndcgAt>`_.

        Return:
            float: nDCG at k (min=0, max=1).
        """
        return self._metrics.ndcgAt(self.k)

    def map(self):
        """Get mean average precision.

        Return:
            float: MAP (min=0, max=1).
        """
        return self._metrics.meanAveragePrecision

    def map_at_k(self):
        """Get mean average precision at k.

        Note:
            More details `on the meanAveragePrecision PySpark documentation <http://spark.apache.org/docs/3.0.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.meanAveragePrecision>`_.

        Return:
            float: MAP at k (min=0, max=1).
        """
        return self._metrics.meanAveragePrecisionAt(self.k)


def _get_top_k_items(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=DEFAULT_K,
):
    """Get the input customer-item-rating tuple in the format of Spark
    DataFrame, output a Spark DataFrame in the dense format of top k items
    for each user.

    Note:
        if it is implicit rating, just append a column of constants to be ratings.

    Args:
        dataframe (pyspark.sql.DataFrame): DataFrame of rating data (in the format of
        customerID-itemID-rating tuple).
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        k (int): number of items for each user.

    Return:
        pyspark.sql.DataFrame: DataFrame of top k items for each user.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_rating).desc())

    # this does not work for rating of the same value.
    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .groupby(col_user)
        .agg(F.collect_list(col_item).alias(col_prediction))
    )

    return items_for_user


def _get_relevant_items_by_threshold(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    threshold=DEFAULT_THRESHOLD,
):
    """Get relevant items for each customer in the input rating data.

    Relevant items are defined as those having ratings above certain threshold.
    The threshold is defined as a statistical measure of the ratings for a
    user, e.g., median.

    Args:
        dataframe: Spark DataFrame of customerID-itemID-rating tuples.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        threshold (float): threshold for determining the relevant recommended items.
            This is used for the case that predicted ratings follow a known
            distribution.

    Return:
        pyspark.sql.DataFrame: DataFrame of customerID-itemID-rating tuples with only relevant
        items.
    """
    items_for_user = (
        dataframe.orderBy(col_rating, ascending=False)
        .where(col_rating + " >= " + str(threshold))
        .select(col_user, col_item, col_rating)
        .withColumn(
            col_prediction, F.collect_list(col_item).over(Window.partitionBy(col_user))
        )
        .select(col_user, col_prediction)
        .dropDuplicates()
    )

    return items_for_user


def _get_relevant_items_by_timestamp(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=DEFAULT_K,
):
    """Get relevant items for each customer defined by timestamp.

    Relevant items are defined as k items that appear mostly recently
    according to timestamps.

    Args:
        dataframe (pyspark.sql.DataFrame): A Spark DataFrame of customerID-itemID-rating-timeStamp
            tuples.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_timestamp (str): column name for timestamp.
        col_prediction (str): column name for prediction.
        k: number of relevant items to be filtered by the function.

    Return:
        pyspark.sql.DataFrame: DataFrame of customerID-itemID-rating tuples with only relevant items.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_timestamp).desc())

    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .withColumn(
            col_prediction, F.collect_list(col_item).over(Window.partitionBy(col_user))
        )
        .select(col_user, col_prediction)
        .dropDuplicates([col_user, col_prediction])
    )

    return items_for_user


class SparkDiversityEvaluation:
    """Spark Evaluator for diversity, coverage, novelty, serendipity"""

    def __init__(
        self,
        train_df,
        reco_df,
        item_feature_df=None,
        item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_relevance=None,
    ):
        """Initializer.

        This is the Spark version of diversity metrics evaluator.
        The methods of this class calculate the following diversity metrics:

        * Coverage - it includes two metrics:
            1. catalog_coverage, which measures the proportion of items that get recommended from the item catalog;
            2. distributional_coverage, which measures how unequally different items are recommended in the
               recommendations to all users.
        * Novelty - A more novel item indicates it is less popular, i.e. it gets recommended less frequently.
        * Diversity - The dissimilarity of items being recommended.
        * Serendipity - The "unusualness" or "surprise" of recommendations to a user. When 'col_relevance' is used,
            it indicates how "pleasant surprise" of recommendations is to a user.

        The metric definitions/formulations are based on the following references with modification:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist: introducing
            serendipity into music recommendation, WSDM 2012

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

            Eugene Yan, Serendipity: Accuracy's unpopular best friend in Recommender Systems,
            eugeneyan.com, April 2020

        Args:
            train_df (pyspark.sql.DataFrame): Data set with historical data for users and items they
                have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
                Interaction here follows the *item choice model* from Castells et al.
            reco_df (pyspark.sql.DataFrame): Recommender's prediction output, containing col_user, col_item,
                col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
            item_feature_df (pyspark.sql.DataFrame): (Optional) It is required only when item_sim_measure='item_feature_vector'.
                It contains two columns: col_item and features (a feature vector).
            item_sim_measure (str): (Optional) This column indicates which item similarity measure to be used.
                Available measures include item_cooccurrence_count (default choice) and item_feature_vector.
            col_user (str): User id column name.
            col_item (str): Item id column name.
            col_relevance (str): Optional. This column indicates whether the recommended item is actually
                relevant to the user or not.
        """

        self.train_df = train_df.select(col_user, col_item)
        self.col_user = col_user
        self.col_item = col_item
        self.sim_col = DEFAULT_SIMILARITY_COL
        self.df_cosine_similarity = None
        self.df_user_item_serendipity = None
        self.df_user_serendipity = None
        self.avg_serendipity = None
        self.df_item_novelty = None
        self.avg_novelty = None
        self.df_intralist_similarity = None
        self.df_user_diversity = None
        self.avg_diversity = None
        self.item_feature_df = item_feature_df
        self.item_sim_measure = item_sim_measure

        if col_relevance is None:
            self.col_relevance = DEFAULT_RELEVANCE_COL
            # relevance term, default is 1 (relevant) for all
            self.reco_df = reco_df.select(
                col_user, col_item, F.lit(1.0).alias(self.col_relevance)
            )
        else:
            self.col_relevance = col_relevance
            self.reco_df = reco_df.select(
                col_user, col_item, F.col(self.col_relevance).cast(DoubleType())
            )

        if self.item_sim_measure == "item_feature_vector":
            self.col_item_features = DEFAULT_ITEM_FEATURES_COL
            required_schema = StructType(
                (
                    StructField(self.col_item, IntegerType()),
                    StructField(self.col_item_features, VectorUDT()),
                )
            )
            if self.item_feature_df is not None:
                if str(required_schema) != str(item_feature_df.schema):
                    raise Exception(
                        "Incorrect schema! item_feature_df should have schema "
                        f"{str(required_schema)} but have {str(item_feature_df.schema)}"
                    )
            else:
                raise Exception(
                    "item_feature_df not specified! item_feature_df must be provided "
                    "if choosing to use item_feature_vector to calculate item similarity. "
                    f"item_feature_df should have schema {str(required_schema)}"
                )

        # check if reco_df contains any user_item pairs that are already shown in train_df
        count_intersection = (
            self.train_df.select(self.col_user, self.col_item)
            .intersect(self.reco_df.select(self.col_user, self.col_item))
            .count()
        )

        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown in train_df"
            )

    def _get_pairwise_items(self, df):
        """Get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])"""
        return (
            df.select(self.col_user, F.col(self.col_item).alias("i1"))
            .join(
                df.select(
                    F.col(self.col_user).alias("_user"),
                    F.col(self.col_item).alias("i2"),
                ),
                (F.col(self.col_user) == F.col("_user")) & (F.col("i1") <= F.col("i2")),
            )
            .select(self.col_user, "i1", "i2")
        )

    def _get_cosine_similarity(self, n_partitions=200):
        if self.item_sim_measure == "item_cooccurrence_count":
            # calculate item-item similarity based on item co-occurrence count
            self._get_cooccurrence_similarity(n_partitions)
        elif self.item_sim_measure == "item_feature_vector":
            # calculate item-item similarity based on item feature vectors
            self._get_item_feature_similarity(n_partitions)
        else:
            raise Exception(
                "item_sim_measure not recognized! The available options include 'item_cooccurrence_count' and 'item_feature_vector'."
            )
        return self.df_cosine_similarity

    def _get_cooccurrence_similarity(self, n_partitions):
        """Cosine similarity metric from

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

        The item indexes in the result are such that i1 <= i2.
        """
        if self.df_cosine_similarity is None:
            pairs = self._get_pairwise_items(df=self.train_df)
            item_count = self.train_df.groupBy(self.col_item).count()

            self.df_cosine_similarity = (
                pairs.groupBy("i1", "i2")
                .count()
                .join(
                    item_count.select(
                        F.col(self.col_item).alias("i1"),
                        F.pow(F.col("count"), 0.5).alias("i1_sqrt_count"),
                    ),
                    on="i1",
                )
                .join(
                    item_count.select(
                        F.col(self.col_item).alias("i2"),
                        F.pow(F.col("count"), 0.5).alias("i2_sqrt_count"),
                    ),
                    on="i2",
                )
                .select(
                    "i1",
                    "i2",
                    (
                        F.col("count")
                        / (F.col("i1_sqrt_count") * F.col("i2_sqrt_count"))
                    ).alias(self.sim_col),
                )
                .repartition(n_partitions, "i1", "i2")
            )
        return self.df_cosine_similarity

    @staticmethod
    @udf(returnType=DoubleType())
    def sim_cos(v1, v2):
        p = 2
        return float(v1.dot(v2)) / float(v1.norm(p) * v2.norm(p))

    def _get_item_feature_similarity(self, n_partitions):
        """Cosine similarity metric based on item feature vectors

        The item indexes in the result are such that i1 <= i2.
        """
        if self.df_cosine_similarity is None:
            self.df_cosine_similarity = (
                self.item_feature_df.select(
                    F.col(self.col_item).alias("i1"),
                    F.col(self.col_item_features).alias("f1"),
                )
                .join(
                    self.item_feature_df.select(
                        F.col(self.col_item).alias("i2"),
                        F.col(self.col_item_features).alias("f2"),
                    ),
                    (F.col("i1") <= F.col("i2")),
                )
                .select("i1", "i2", self.sim_cos("f1", "f2").alias("sim"))
                .sort("i1", "i2")
                .repartition(n_partitions, "i1", "i2")
            )
        return self.df_cosine_similarity

    # Diversity metrics
    def _get_intralist_similarity(self, df):
        """Intra-list similarity from

        :Citation:

            "Improving Recommendation Lists Through Topic Diversification",
            Ziegler, McNee, Konstan and Lausen, 2005.
        """
        if self.df_intralist_similarity is None:
            pairs = self._get_pairwise_items(df=df)
            similarity_df = self._get_cosine_similarity()
            # Fillna(0) is needed in the cases where similarity_df does not have an entry for a pair of items.
            # e.g. i1 and i2 have never occurred together.
            self.df_intralist_similarity = (
                pairs.join(similarity_df, on=["i1", "i2"], how="left")
                .fillna(0)
                .filter(F.col("i1") != F.col("i2"))
                .groupBy(self.col_user)
                .agg(F.mean(self.sim_col).alias("avg_il_sim"))
                .select(self.col_user, "avg_il_sim")
            )
        return self.df_intralist_similarity

    def user_diversity(self):
        """Calculate average diversity of recommendations for each user.
        The metric definition is based on formula (3) in the following reference:

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with the following columns: col_user, user_diversity.
        """
        if self.df_user_diversity is None:
            self.df_intralist_similarity = self._get_intralist_similarity(self.reco_df)
            self.df_user_diversity = (
                self.df_intralist_similarity.withColumn(
                    "user_diversity", 1 - F.col("avg_il_sim")
                )
                .select(self.col_user, "user_diversity")
                .orderBy(self.col_user)
            )
        return self.df_user_diversity

    def diversity(self):
        """Calculate average diversity of recommendations across all users.

        Returns:
            float: diversity.
        """
        if self.avg_diversity is None:
            self.df_user_diversity = self.user_diversity()
            self.avg_diversity = self.df_user_diversity.agg(
                {"user_diversity": "mean"}
            ).first()[0]
        return self.avg_diversity

    # Novelty metrics
    def historical_item_novelty(self):
        """Calculate novelty for each item. Novelty is computed as the minus logarithm of
        (number of interactions with item / total number of interactions). The definition of the metric
        is based on the following reference using the choice model (eqs. 1 and 6):

        :Citation:

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

        The novelty of an item can be defined relative to a set of observed events on the set of all items.
        These can be events of user choice (item "is picked" by a random user) or user discovery
        (item "is known" to a random user). The above definition of novelty reflects a factor of item popularity.
        High novelty values correspond to long-tail items in the density function, that few users have interacted
        with and low novelty values correspond to popular head items.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with the following columns: col_item, item_novelty.
        """
        if self.df_item_novelty is None:
            n_records = self.train_df.count()
            self.df_item_novelty = (
                self.train_df.groupBy(self.col_item)
                .count()
                .withColumn("item_novelty", -F.log2(F.col("count") / n_records))
                .select(self.col_item, "item_novelty")
                .orderBy(self.col_item)
            )
        return self.df_item_novelty

    def novelty(self):
        """Calculate the average novelty in a list of recommended items (this assumes that the recommendation list
        is already computed). Follows section 5 from

        :Citation:

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: novelty.
        """
        if self.avg_novelty is None:
            self.df_item_novelty = self.historical_item_novelty()
            n_recommendations = self.reco_df.count()
            self.avg_novelty = (
                self.reco_df.groupBy(self.col_item)
                .count()
                .join(self.df_item_novelty, self.col_item)
                .selectExpr("sum(count * item_novelty)")
                .first()[0]
                / n_recommendations
            )
        return self.avg_novelty

    # Serendipity metrics
    def user_item_serendipity(self):
        """Calculate serendipity of each item in the recommendations for each user.
        The metric definition is based on the following references:

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

            Eugene Yan, Serendipity: Accuracy’s unpopular best friend in Recommender Systems,
            eugeneyan.com, April 2020

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with columns: col_user, col_item, user_item_serendipity.
        """
        # for every col_user, col_item in reco_df, join all interacted items from train_df.
        # These interacted items are repeated for each item in reco_df for a specific user.
        if self.df_user_item_serendipity is None:
            self.df_cosine_similarity = self._get_cosine_similarity()
            self.df_user_item_serendipity = (
                self.reco_df.select(
                    self.col_user,
                    self.col_item,
                    F.col(self.col_item).alias(
                        "reco_item_tmp"
                    ),  # duplicate col_item to keep
                )
                .join(
                    self.train_df.select(
                        self.col_user, F.col(self.col_item).alias("train_item_tmp")
                    ),
                    on=[self.col_user],
                )
                .select(
                    self.col_user,
                    self.col_item,
                    F.least(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i1"
                    ),
                    F.greatest(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i2"
                    ),
                )
                .join(self.df_cosine_similarity, on=["i1", "i2"], how="left")
                .fillna(0)
                .groupBy(self.col_user, self.col_item)
                .agg(F.mean(self.sim_col).alias("avg_item2interactedHistory_sim"))
                .join(self.reco_df, on=[self.col_user, self.col_item])
                .withColumn(
                    "user_item_serendipity",
                    (1 - F.col("avg_item2interactedHistory_sim"))
                    * F.col(self.col_relevance),
                )
                .select(self.col_user, self.col_item, "user_item_serendipity")
                .orderBy(self.col_user, self.col_item)
            )
        return self.df_user_item_serendipity

    def user_serendipity(self):
        """Calculate average serendipity for each user's recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_user, user_serendipity.
        """
        if self.df_user_serendipity is None:
            self.df_user_item_serendipity = self.user_item_serendipity()
            self.df_user_serendipity = (
                self.df_user_item_serendipity.groupBy(self.col_user)
                .agg(F.mean("user_item_serendipity").alias("user_serendipity"))
                .orderBy(self.col_user)
            )
        return self.df_user_serendipity

    def serendipity(self):
        """Calculate average serendipity for recommendations across all users.

        Returns:
            float: serendipity.
        """
        if self.avg_serendipity is None:
            self.df_user_serendipity = self.user_serendipity()
            self.avg_serendipity = self.df_user_serendipity.agg(
                {"user_serendipity": "mean"}
            ).first()[0]
        return self.avg_serendipity

    # Coverage metrics
    def catalog_coverage(self):
        """Calculate catalog coverage for recommendations across all users.
        The metric definition is based on the "catalog coverage" definition in the following reference:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

        Returns:
            float: catalog coverage
        """
        # distinct item count in reco_df
        count_distinct_item_reco = self.reco_df.select(self.col_item).distinct().count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.col_item).distinct().count()
        )

        # catalog coverage
        c_coverage = count_distinct_item_reco / count_distinct_item_train
        return c_coverage

    def distributional_coverage(self):
        """Calculate distributional coverage for recommendations across all users.
        The metric definition is based on formula (21) in the following reference:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

        Returns:
            float: distributional coverage
        """
        # In reco_df, how  many times each col_item is being recommended
        df_itemcnt_reco = self.reco_df.groupBy(self.col_item).count()

        # the number of total recommendations
        count_row_reco = self.reco_df.count()
        df_entropy = df_itemcnt_reco.withColumn(
            "p(i)", F.col("count") / count_row_reco
        ).withColumn("entropy(i)", F.col("p(i)") * F.log2(F.col("p(i)")))
        # distributional coverage
        d_coverage = -df_entropy.agg(F.sum("entropy(i)")).collect()[0][0]

        return d_coverage

# COMMAND ----------

# from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k

# COMMAND ----------

# selected_df_1 = predictions_prob_1.select('asin1', 'asin2', 'prob')
predictions = spark.read.table("asac.pred_1_we")
test_df = spark.sql("SELECT * FROM asac.240430_test_df")
test_df = test_df.drop('target')

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df, 
                                   rating_pred= predictions, 
                                   k = 5, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.5)


print("Model: {}".format('GBDT(by threshold 0.5)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.3)


print("Model: {}".format('GBDT(by threshold 0.3)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

import pyspark.sql.functions as F
 
test_df2 = test_df.filter(F.col('review_cnts') >= 2)

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 5, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df2, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.5)


print("Model: {}".format('GBDT(by threshold 0.5)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df2, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.3)


print("Model: {}".format('GBDT(by threshold 0.3)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 6번 모델 랭킹평가 진행

# COMMAND ----------

predictions = spark.read.table("asac.pred_6")
test_df = spark.sql("SELECT * FROM asac.240430_test_df")
test_df = test_df.drop('target')

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df, 
                                   rating_pred= predictions, 
                                   k = 5, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.5)


print("Model: {}".format('GBDT(by threshold 0.5)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.3)


print("Model: {}".format('GBDT(by threshold 0.3)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

import pyspark.sql.functions as F
 
test_df2 = test_df.filter(F.col('review_cnts') >= 2)

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 5, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k')


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df2, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.5)


print("Model: {}".format('GBDT(by threshold 0.5)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------

rank_eval2 = SparkRankingEvaluation(rating_true = test_df2, 
                                    rating_pred= predictions, 
                                    k = 10, col_user='asin1', col_item='asin2', 
                                    col_rating='review_cnts', col_prediction='prob', 
                                    relevancy_method='by_threshold', threshold=0.3)


print("Model: {}".format('GBDT(by threshold 0.3)'), "Top K:%d" % rank_eval2.k, "MAP:%f" % rank_eval2.map_at_k(), "NDCG:%f" % rank_eval2.ndcg_at_k(), "Precision@K:%f" % rank_eval2.precision_at_k(), "Recall@K:%f" % rank_eval2.recall_at_k())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### df_5_cos (10만개 train 데이터에 변수 붙인거) 변수 분포 확인하기

# COMMAND ----------

df_5_cos = spark.read.table("asac.df_5_cos")

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

columns = ['mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']
df_selected = df_5_cos.select(*columns)

# Spark DataFrame을 pandas DataFrame으로 변환
pdf = df_selected.toPandas()

# COMMAND ----------

# Density plot 그리기
plt.figure(figsize=(10, 8))
for column in pdf.columns:
    sns.kdeplot(pdf[column], shade=True, label=column)
plt.title('Density Plots')
plt.legend()
plt.show()

# COMMAND ----------

plt.figure(figsize=(13,10))
for column in pdf.columns:
    sns.kdeplot(pdf[column], shade=True, label=column)

plt.title('Density Plots')
plt.legend()

plt.xlim(0, 1.1)

plt.show()

# COMMAND ----------

# 'target' 컬럼 포함하여 선택
columns_with_target = columns + ['target']
df_with_target = df_5_cos.select(*columns_with_target)

# pandas DataFrame으로 변환
pdf_with_target = df_with_target.toPandas()

# 'target' 값에 따른 density plot 그리기
columns_to_plot = ['std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']
for column in columns_to_plot:
    plt.figure(figsize=(10,8))
    sns.kdeplot(pdf_with_target[pdf_with_target['target'] == 0][column], shade=True, label='Target 0')
    sns.kdeplot(pdf_with_target[pdf_with_target['target'] == 1][column], shade=True, label='Target 1')
    plt.title(f'Density Plot of {column} by Target')
    plt.xlim(0, 1.1)
    plt.legend()
    plt.show()


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 4백만개 train 그래프 그려보기 (text, image 둘다 없는건 제외한 comb 데이터로)

# COMMAND ----------

total_train_df = spark.read.table("asac.total_train_temp_fin")
text = spark.read.table("asac.total_train_temp_fin_text")
image = spark.read.table("asac.total_train_temp_fin_image")
comb = spark.read.table("asac.total_train_temp_fin_text_image")

# COMMAND ----------

columns = ['mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']
df_selected = comb.select(*columns)

# Spark DataFrame을 pandas DataFrame으로 변환
pdf = df_selected.toPandas()

# COMMAND ----------

# Density plot 그리기
plt.figure(figsize=(10, 8))
for column in pdf.columns:
    sns.kdeplot(pdf[column], shade=True, label=column)
plt.title('Density Plots')
plt.legend()
plt.show()

# COMMAND ----------

plt.figure(figsize=(13,10))
for column in pdf.columns:
    sns.kdeplot(pdf[column], shade=True, label=column)

plt.title('Density Plots')
plt.legend()
plt.xlim(-0.5, 1.1)

plt.show()

# COMMAND ----------

# 'target' 컬럼 포함하여 선택
columns_with_target = columns + ['target']
df_with_target = comb.select(*columns_with_target)

# pandas DataFrame으로 변환
pdf_with_target = df_with_target.toPandas()

# 'target' 값에 따른 density plot 그리기
columns_to_plot = ['mean_vector_cosine_similarity','std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']
for column in columns_to_plot:
    plt.figure(figsize=(10,8))
    sns.kdeplot(pdf_with_target[pdf_with_target['target'] == 0][column], shade=True, label='Target 0')
    sns.kdeplot(pdf_with_target[pdf_with_target['target'] == 1][column], shade=True, label='Target 1')
    plt.title(f'Density Plot of {column} by Target')
    plt.xlim(-0.5, 1.1)
    plt.legend()
    plt.show()


# COMMAND ----------


