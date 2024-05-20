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

df_text.count()

# COMMAND ----------

display(df_text)

# COMMAND ----------

display(df_5)

# COMMAND ----------

col_names = df_5.columns
col_names

# COMMAND ----------

# MAGIC %md
# MAGIC ### 이미지 임베딩 전처리
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


df_5.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## df_5 컬럼 합치기

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



# COMMAND ----------

display(df_5.limit(10))

# COMMAND ----------

from pyspark.sql.functions import col

df_5 = df_5.join(df_image.alias("image1"), df_5.asin1 == col("image1.asin"), "left_outer")
df_5 = df_5.withColumnRenamed("embedding_array", "image_1")
df_5 = df_5.drop("asin")

df_5 = df_5.join(df_image.alias("image2"), df_5.asin2 == col("image2.asin"), "left_outer")
df_5 = df_5.withColumnRenamed("embedding_array", "image_2")
df_5 = df_5.drop("asin")

# COMMAND ----------

display(df_5.limit(10))

# COMMAND ----------

df_5_cos = df_5

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

total_col = df_5_cos.columns
total_col

# COMMAND ----------

# 테이블 저장하기
name = "asac.df_5_cos"
df_5_cos.write.saveAsTable(name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 부스팅 모델링 (Gradient-Boosted Tree (GBT))
# MAGIC 1. 이미지 임베딩 - 텍스트 통계량 임베딩
# MAGIC 2. 이미지 임베딩 - 텍스트 통계량 유사도
# MAGIC 3. 이미지 임베딩 - 텍스트 통계량 평균유사도
# MAGIC 4. 이미지 유사도 - 텍스트 통계량 임베딩
# MAGIC 5. 이미지 유사도 - 텍스트 통계량 유사도
# MAGIC 6. 이미지 유사도 - 텍스트 통계량 평균유사도
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# 데이터 준비 및 전처리
vectorAssembler_1 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
vectorAssembler_2 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
vectorAssembler_3 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
vectorAssembler_4 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
vectorAssembler_5 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
vectorAssembler_6 = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")

# COMMAND ----------

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler, gbt])
model = pipeline.fit(trainingData)

# COMMAND ----------

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions = model.transform(testData)

# COMMAND ----------

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recall")

precision = evaluatorPrecision.evaluate(predictions)
recall = evaluatorRecall.evaluate(predictions)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


