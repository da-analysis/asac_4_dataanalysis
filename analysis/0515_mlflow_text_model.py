# Databricks notebook source
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
import pyspark.pandas as ps
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings, AlbertEmbeddings, Tokenizer, Normalizer, StopWordsCleaner, RoBertaSentenceEmbeddings, Doc2VecModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.functions import col, size, array, expr
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType

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

# MAGIC %md
# MAGIC ### 10만개 train에 맞게 전처리한 데이터
# MAGIC - asac.df_5_cos

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin1, asin2, mean_vector_cosine_similarity from asac.df_5_cos
# MAGIC limit 10

# COMMAND ----------

df_5_cos = spark.read.table("asac.df_5_cos")

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

from pyspark.sql.functions import col, count, lit
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

# COMMAND ----------

# GBT 모델 설정 및 학습
vectorAssembler_1 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity'], outputCol="features")

train_df_1 = train_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()

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


# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
test_df_1 = test_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
test_df_1 = test_df_1.na.drop()
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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


