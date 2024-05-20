# Databricks notebook source
# MAGIC %md
# MAGIC ### 6번에서 쓴 데이터 써서 1-1번 돌려보기
# MAGIC - 만약, 이게 6번보다 더 좋은 성능 가진다면, 1-1번만 변수 중요도 확인하기
# MAGIC - 6번이 더 좋게 나온다면, 1-1과 6번 그리고 데이터 바꾼 1-1 번 변수중요도 확인하기
# MAGIC - 모델 선정 후에는 파라미터 튜일하기
# MAGIC - 확률분포도 확인하기, 랭킹모델 만들기
# MAGIC

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

# COMMAND ----------

# 텍스트
df_text = spark.read.table("asac.embed_cell_sbert_32_fin")

# 5배수
df_5 = spark.read.table("asac.240430_review3_5multiply")

# COMMAND ----------

## 완전 전체 train 셋과 test 셋
total_train = spark.read.table("asac.240430_train_df")
total_test =  spark.read.table("asac.240430_test_df")

# COMMAND ----------

df_5_cos = spark.read.table("asac.df_5_cos")

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin1, image_1, image_2, image_cosine_similarity from asac.df_5_cos
# MAGIC limit 10

# COMMAND ----------

df_5_cos.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6번(텍스트 통계량 유사도 & 이미지 유사도) 데이터로 텍스트 통계량 유사도 변수만 활용해서 모델 확인하기

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

from pyspark.sql.functions import when

train_df_6 = train_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
train_df_6 = train_df_6.na.drop()
test_df_6 = test_df_to.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
test_df_6 = test_df_6.na.drop()

vectorAssembler_1 = VectorAssembler(inputCols=['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity'], outputCol="features")


# 불균형 가중치 
numPositives = train_df_6.filter(train_df_6["target"] == 1).count()
numNegatives = train_df_6.filter(train_df_6["target"] == 0).count()
total = train_df_6.count()

balanceRatio = numNegatives / total

train_df_6 = train_df_6.withColumn('classWeight', when(train_df_6['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# GBT 모델 설정 및 학습
gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_1, gbt])
model_6 = pipeline.fit(train_df_6)

# 테스트 데이터에 대한 예측 수행 (train 안에 있는거, 일종의 valid)
predictions_6 = model_6.transform(test_df_6)

# COMMAND ----------

model_path_new = "dbfs:/FileStore/amazon/model/model_new"
model_6.write().overwrite().save(model_path_new)

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

# GBT 모델에서 변수 중요도 추출
featureImportances = model_6.stages[-1].featureImportances

# 입력 변수 목록
inputCols = ['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 
             'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 
             'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 
             'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 
             'skewness_cosine_similarity', 'kurtosis_cosine_similarity']

# 변수 중요도와 변수 이름을 매핑
importances = {inputCols[i]: featureImportances[i] for i in range(len(inputCols))}

# 중요도에 따라 변수 정렬
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

import matplotlib.pyplot as plt
import seaborn as sns

# 변수 이름과 중요도를 분리하여 리스트로 저장
names, values = zip(*sorted_importances)

# 시각화
plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 원래 6번 데이터 및 6번 모델링

# COMMAND ----------

from pyspark.ml import PipelineModel

# 모델 저장 경로
model_path_6 = "dbfs:/FileStore/amazon/model/model_6"

# 저장된 모델 불러오기
loaded_model_6 = PipelineModel.load(model_path_6)

# 테스트 데이터에 대한 예측 수행
predictions_6 = loaded_model_6.transform(test_df_6)

# GBT 모델 추출
gbt_model = loaded_model_6.stages[-1]

# 변수 중요도 가져오기
feature_importances = gbt_model.featureImportances

# 입력 변수 목록
input_cols = ['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 
              'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 
              'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 
              'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 
              'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']

# 중요도와 변수 이름을 매핑
importances_with_names = [(input_cols[i], feature_importances[i]) for i in range(len(input_cols))]

# 중요도 순으로 정렬
sorted_importances = sorted(importances_with_names, key=lambda x: x[1], reverse=True)

# 변수 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns

names, values = zip(*sorted_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())

predictions_prob_6 = predictions_6.withColumn("prob", extract_prob_udf(col("probability")))

#predictions.createOrReplaceTempView("predictions")  

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# PySpark DataFrame을 Pandas DataFrame으로 변환
pandas_df = predictions_prob_6.select("prob").toPandas()
# Seaborn을 사용하여 확률 분포표 그리기
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['prob'], bins=20, kde=True)
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark DataFrame을 Pandas DataFrame으로 변환
pandas_df = predictions_prob_6.select("prob").toPandas()

# Convert pandas_df['prob'] to numpy array
prob_array = pandas_df['prob'].values

# Seaborn을 사용하여 확률 분포표 그리기
plt.figure(figsize=(10, 6))
sns.histplot(prob_array, bins=20, kde=True)
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

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
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 원래 1-1번 모델 & 해당데이터

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


from pyspark.sql.functions import when
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, count, lit

train_df_1 = train_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()


test_df_1 = test_df.select(['cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
test_df_1 = test_df_1.na.drop()

# 불균형 가중치 
numPositives = train_df_1.filter(train_df_1["target"] == 1).count()
numNegatives = train_df_1.filter(train_df_1["target"] == 0).count()
total = train_df_1.count()

balanceRatio = numNegatives / total

train_df_1 = train_df_1.withColumn('classWeight', when(train_df_1['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# COMMAND ----------

from pyspark.ml import PipelineModel

# 모델 저장 경로
model_path_1_we = "dbfs:/FileStore/amazon/model/model_1_we"

# 저장된 모델 불러오기
loaded_model_path_1_we = PipelineModel.load(model_path_1_we)

# COMMAND ----------

# 테스트 데이터에 대한 예측 수행
predictions_1_we = loaded_model_path_1_we.transform(test_df_1)

# GBT 모델 추출
gbt_model = loaded_model_path_1_we.stages[-1]

# 변수 중요도 가져오기
feature_importances = gbt_model.featureImportances

# 입력 변수 목록
input_cols = ['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 
              'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 
              'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 
              'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 
              'skewness_cosine_similarity', 'kurtosis_cosine_similarity']

# 중요도와 변수 이름을 매핑
importances_with_names = [(input_cols[i], feature_importances[i]) for i in range(len(input_cols))]

# 중요도 순으로 정렬
sorted_importances = sorted(importances_with_names, key=lambda x: x[1], reverse=True)

# 변수 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns

names, values = zip(*sorted_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql.functions import col

# 결과 확인
predictions_1_we.show(1)

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
def extract_prob(v):
    try:
        return float(v[1])  # Your VectorUDT is of length 2
    except ValueError:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())

predictions_prob_1 = predictions_1_we.withColumn("prob", extract_prob_udf(col("probability")))

#predictions.createOrReplaceTempView("predictions")  

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# PySpark DataFrame을 Pandas DataFrame으로 변환
pandas_df = predictions_prob_1.select("prob").toPandas()
# Seaborn을 사용하여 확률 분포표 그리기
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['prob'], bins=20, kde=True)
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark DataFrame을 Pandas DataFrame으로 변환
pandas_df = predictions_prob_1.select("prob").toPandas()

# Convert pandas_df['prob'] to numpy array
prob_array = pandas_df['prob'].values

# Seaborn을 사용하여 확률 분포표 그리기
plt.figure(figsize=(10, 6))
sns.histplot(prob_array, bins=20, kde=True)
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

prob_list = predictions_prob_1.select("prob").rdd.flatMap(lambda x: x).collect()

# 리스트를 numpy 배열로 변환
prob_array = np.array(prob_list)

# matplotlib를 이용한 시각화
plt.figure(figsize=(10, 6))
plt.hist(prob_array, bins=20, density=True, alpha=0.6, color='b')
plt.title('Class 1 Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 파라미터 튜닝 진행

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import when

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

# 텍스트 없는 null 값가진 행제거한 데이터 셋
df_5_cos_null_text = df_5_cos.na.drop(subset=["new_pcaValues32_1"])
df_5_cos_null_text = df_5_cos_null_text.select(['asin1','asin2','cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'target', 'new_pcaValues32_1', 'list_length_1', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'new_pcaValues32_2', 'list_length_2', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2',
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


from pyspark.sql.functions import when
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, count, lit

train_df_1 = train_df.select(['asin1','asin2','cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()


test_df_1 = test_df.select(['asin1','asin2','cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
test_df_1 = test_df_1.na.drop()

# 불균형 가중치 
numPositives = train_df_1.filter(train_df_1["target"] == 1).count()
numNegatives = train_df_1.filter(train_df_1["target"] == 0).count()
total = train_df_1.count()

balanceRatio = numNegatives / total

train_df_1 = train_df_1.withColumn('classWeight', when(train_df_1['target'] == 1, balanceRatio).otherwise(1 - balanceRatio))

# COMMAND ----------

## 1-1 번 모델
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10)
pipeline = Pipeline(stages=[vectorAssembler_1, gbt])
#model = pipeline.fit(train_df_1)

# COMMAND ----------

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10, 15, 20]) \
    .addGrid(gbt.maxBins, [32, 64, 96]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="target")

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)  # 5-fold cross-validation

# 교차 검증을 사용하여 모델 학습
cvModel = cv.fit(train_df_1)

# 최적의 모델을 테스트 데이터셋에 적용
predictions_1_par = cvModel.transform(test_df_1)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

bestModel_1 = cvModel.bestModel
bestModelPath_1 = "dbfs:/FileStore/amazon/model/best_param_1"
bestModel_1.save(bestModelPath_1)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_1_par)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_1_par)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_1_par)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_1_par)
recall = evaluatorRecall.evaluate(predictions_1_par)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

from pyspark.ml import PipelineModel

# 모델 저장 경로
bestModelPath_1 = "dbfs:/FileStore/amazon/model/best_param_1"

# 저장된 모델 불러오기
bestModel_1 = PipelineModel.load(bestModelPath_1)

# 테스트 데이터에 대한 예측 수행
predictions_1 = bestModel_1.transform(test_df_1)

# GBT 모델 추출
gbt_model = bestModel_1.stages[-1]

# 변수 중요도 가져오기
feature_importances = gbt_model.featureImportances

# 입력 변수 목록
inputCols = ['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 
             'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 
             'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 
             'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 
             'skewness_cosine_similarity', 'kurtosis_cosine_similarity']

# 중요도와 변수 이름을 매핑
importances_with_names = [(input_cols[i], feature_importances[i]) for i in range(len(input_cols))]

# 중요도 순으로 정렬
sorted_importances = sorted(importances_with_names, key=lambda x: x[1], reverse=True)

# 변수 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns

names, values = zip(*sorted_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# COMMAND ----------



# COMMAND ----------

## 6번 모델
df_total = df_5_cos.select(['asin1','asin2'
,    'cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'target', 'new_pcaValues32_1', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'new_pcaValues32_2', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity', 'cosine_fin'])
df_total = df_total.na.drop(subset=["image_1","image_2","new_pcaValues32_1","new_pcaValues32_2"])

# train 데이터와 test 데이터 나누기
fractions = df_total.select("target").distinct().rdd.flatMap(lambda x: x).collect()
fractions = {row: 0.8 for row in fractions}  # 트레인셋 80%

# `sampleBy` 함수를 사용하여 트레인셋 추출
train_df_to = df_total.sampleBy("target", fractions, seed=42)

# `exceptAll`을 이용해서 트레인셋에 없는 행들을 테스트셋으로 설정
test_df_to = df_total.exceptAll(train_df_to)

train_df_6 = train_df_to.select(['asin1','asin2','cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
'target'])
train_df_6 = train_df_6.na.drop()
test_df_6 = test_df_to.select(['asin1','asin2','cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
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

# COMMAND ----------

## 5로 안하고 3으로 k-fold 진행
## 파라미터 경우의 수도 줄임
## maxMemoryInMB=512로 설정함

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

gbt = GBTClassifier(labelCol="target", featuresCol="features", weightCol="classWeight",maxIter=10, maxMemoryInMB=512)
pipeline = Pipeline(stages=[vectorAssembler_6, gbt])
#model = pipeline.fit(train_df_6)


paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10, 15]) \
    .addGrid(gbt.maxBins, [32, 64]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="target")

cv_6 = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3)  # 3-fold cross-validation

# 교차 검증을 사용하여 모델 학습
cvModel_6 = cv_6.fit(train_df_6)

# 최적의 모델을 테스트 데이터셋에 적용
predictions_6_par = cvModel_6.transform(test_df_6)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

bestModel_6 = cvModel_6.bestModel
bestModelPath_6 = "dbfs:/FileStore/amazon/model/best_param_6"
bestModel_6.save(bestModelPath_6)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# 평가 지표로 모델 성능 평가 (auc)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions_6_par)

# 정확도(Accuracy) 계산
evaluatorAccuracy = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorAccuracy.evaluate(predictions_6_par)

# F1-Score 계산
evaluatorF1 = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
f1Score = evaluatorF1.evaluate(predictions_6_par)

# Precision과 Recall 계산
evaluatorPrecision = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
evaluatorRecall = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")

precision = evaluatorPrecision.evaluate(predictions_6_par)
recall = evaluatorRecall.evaluate(predictions_6_par)

print(f"AUC: {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1Score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# COMMAND ----------

from pyspark.ml import PipelineModel

# 모델 저장 경로
bestModelPath_6 = "dbfs:/FileStore/amazon/model/best_param_6"

# 저장된 모델 불러오기
loaded_model_6 = PipelineModel.load(bestModelPath_6)

# 테스트 데이터에 대한 예측 수행
predictions_6 = loaded_model_6.transform(test_df_6)

# GBT 모델 추출
gbt_model = loaded_model_6.stages[-1]

# 변수 중요도 가져오기
feature_importances = gbt_model.featureImportances

# 입력 변수 목록
input_cols = ['cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 
              'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 
              'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 
              'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 
              'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity']

# 중요도와 변수 이름을 매핑
importances_with_names = [(input_cols[i], feature_importances[i]) for i in range(len(input_cols))]

# 중요도 순으로 정렬
sorted_importances = sorted(importances_with_names, key=lambda x: x[1], reverse=True)

# 변수 중요도 시각화
import matplotlib.pyplot as plt
import seaborn as sns

names, values = zip(*sorted_importances)

plt.figure(figsize=(10, 8))
sns.barplot(x=values, y=names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 전체 train셋(5백만개)에 변수 옆으로 붙이기

# COMMAND ----------

## 완전 전체 train 셋과 test 셋
total_train = spark.read.table("asac.240430_train_df")
total_test =  spark.read.table("asac.240430_test_df")
# 추후 전체 다 진행하고자 할 때는 삭제하기
total_train = total_train.limit(10000)

# COMMAND ----------

# 이미지
# df_image =  spark.read.csv("dbfs:/FileStore/amazon/data/image/AC_image_embedding",header=True)


# 텍스트
df_text = spark.read.table("asac.embed_cell_sbert_32_fin")

# 5배수
df_5 = spark.read.table("asac.240430_review3_5multiply")

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

# asin1과 asin2 를 기준으로 left outer join 진행
# 임베딩, 통계량 값, 길이
col_names = total_train.columns

total_train = total_train.join(df_text, total_train.asin1 == df_text.asin,"left_outer")
total_train = total_train.drop("asin")

for col_name in total_train.columns:
    if col_name not in col_names:  
        total_train = total_train.withColumnRenamed(col_name, col_name + "_1")

col_names = total_train.columns

total_train = total_train.join(df_text, total_train.asin2 == df_text.asin,"left_outer")
total_train = total_train.drop("asin")

for col_name in total_train.columns:
    if col_name not in col_names:  
        total_train = total_train.withColumnRenamed(col_name, col_name + "_2")

total_train = total_train.drop("asin")
total_train = total_train.drop("variance_1")
total_train = total_train.drop("variance_2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 조인과정으로 인해 일단 데이터 저장한 후에 다시 조인 진행

# COMMAND ----------

name = "asac.total_train_10000_temp1"
total_train.write.saveAsTable(name)

# COMMAND ----------

total_train = spark.read.table("asac.total_train_10000_temp1")

# COMMAND ----------

from pyspark.sql.functions import col

total_train = total_train.join(df_image.alias("image1"), total_train.asin1 == col("image1.asin"), "left_outer")
total_train = total_train.withColumnRenamed("embedding_array", "image_1")
total_train = total_train.drop("asin")

total_train = total_train.join(df_image.alias("image2"), total_train.asin2 == col("image2.asin"), "left_outer")
total_train = total_train.withColumnRenamed("embedding_array", "image_2")
total_train = total_train.drop("asin")

# COMMAND ----------

name = "asac.total_train_10000_temp2"
total_train.write.saveAsTable(name)

# COMMAND ----------

total_train_temp = spark.read.table("asac.total_train_10000_temp2")

# COMMAND ----------

new_pcaValues32_1 = total_train_temp.filter(total_train_temp["new_pcaValues32_1"].isNull()).count()
new_pcaValues32_2 = total_train_temp.filter(total_train_temp["new_pcaValues32_2"].isNull()).count()
image_1 = total_train_temp.filter(total_train_temp["image_1"].isNull()).count()
image_2 = total_train_temp.filter(total_train_temp["image_2"].isNull()).count()

print(f"'new_pcaValues32_1' 컬럼의 널 값 개수: {new_pcaValues32_1}")
print(f"'new_pcaValues32_2' 컬럼의 널 값 개수: {new_pcaValues32_2}")
print(f"'image_1' 컬럼의 널 값 개수: {image_1}")
print(f"'image_2' 컬럼의 널 값 개수: {image_2}")

# COMMAND ----------

total_train_temp.show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 코사인 유사도 계산하기

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
    total_train_temp = total_train_temp.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    total_train_temp = total_train_temp.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# COMMAND ----------

# MAGIC %sql
# MAGIC drop table asac.total_train_10000_temp3

# COMMAND ----------

name = "asac.total_train_10000_temp3"
total_train_temp.write.saveAsTable(name)

# COMMAND ----------

total_train_temp_te = spark.read.table("asac.total_train_10000_temp3")

# COMMAND ----------

total_train_temp_te.columns

# COMMAND ----------

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
    total_train_temp_im = total_train_temp_te.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    total_train_temp_im = total_train_temp_im.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# COMMAND ----------

total_train_temp_im.columns

# COMMAND ----------

total_train_temp.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 테이블 저장하기

# COMMAND ----------

name = "asac.total_train_10000_temp_fin"
total_train_temp_im.write.saveAsTable(name)

# COMMAND ----------

total_train_df = spark.read.table("asac.total_train_10000_temp_fin")

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin1, image_1, image_2, image_cosine_similarity from asac.total_train_10000_temp_fin
# MAGIC limit 10  -- 유사도 값이 이상함..

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

train_df_1 = total_train_df.select(['asin1','asin2','review_cnts','cat2_same', 'cat3_same', 'asin1_count_prevMonth','asin2_count_prevMonth', 'asin1_cat2_vec','asin1_cat3_vec','asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity','mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity','q2_cosine_similarity','q3_cosine_similarity','skewness_cosine_similarity','kurtosis_cosine_similarity','target'])
train_df_1 = train_df_1.na.drop()

# COMMAND ----------

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
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

predictions_prob_1.columns

# COMMAND ----------

selected_df_1 = predictions_prob_1.select('asin1', 'asin2', 'prob','review_cnts')
name = "asac.pred_1_we"
selected_df_1.write.saveAsTable(name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 6번 모델로 예측 진행

# COMMAND ----------

train_6 = train_df.select(['asin1','asin2','review_cnts',
    'cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'target', 'new_pcaValues32_1', 'mean_vector_1', 'std_dev_1', 'q1_1', 'q2_1', 'q3_1', 'skewness_1', 'kurtosis_1', 'new_pcaValues32_2', 'mean_vector_2', 'std_dev_2', 'q1_2', 'q2_2', 'q3_2', 'skewness_2', 'kurtosis_2', 'image_1', 'image_2', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity','q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity', 'cosine_fin'])
train_6 = train_6.na.drop(subset=["image_1","image_2","new_pcaValues32_1","new_pcaValues32_2"])

train_df_6 = train_df_to.select(['asin1','asin2','review_cnts','cat2_same', 'cat3_same', 'asin1_count_prevMonth', 'asin2_count_prevMonth', 'asin1_cat2_vec', 'asin1_cat3_vec', 'asin2_cat2_vec', 'asin2_cat3_vec', 'price_similarity', 'mean_vector_cosine_similarity', 'std_dev_cosine_similarity', 'q1_cosine_similarity', 'q2_cosine_similarity', 'q3_cosine_similarity', 'skewness_cosine_similarity', 'kurtosis_cosine_similarity', 'image_cosine_similarity',
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
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

selected_df_6 = predictions_prob_6.select('asin1', 'asin2', 'prob','review_cnts')
name = "asac.pred_6"
selected_df_6.write.saveAsTable(name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-1번 모델 랭킹평가 진행
# MAGIC

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
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

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

rank_eval3 = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)

print("Model: {}".format('GBDT(top k=10, test data co-review >=2)'), "Top K:%d" % rank_eval3.k, "MAP:%f" % rank_eval3.map_at_k(), "NDCG:%f" % rank_eval3.ndcg_at_k(), "Precision@K:%f" % rank_eval3.precision_at_k(), "Recall@K:%f" % rank_eval3.recall_at_k())

# COMMAND ----------

rank_eval4 = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)

print("Model: {}".format('GBDT(top k=10, test data co-review >=2)'), "Top K:%d" % rank_eval4.k, "MAP:%f" % rank_eval4.map_at_k(), "NDCG:%f" % rank_eval4.ndcg_at_k(), "Precision@K:%f" % rank_eval4.precision_at_k(), "Recall@K:%f" % rank_eval4.recall_at_k())

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
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)


print("Model: {}".format('GBDT'), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())

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

rank_eval3 = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)

print("Model: {}".format('GBDT(top k=10, test data co-review >=2)'), "Top K:%d" % rank_eval3.k, "MAP:%f" % rank_eval3.map_at_k(), "NDCG:%f" % rank_eval3.ndcg_at_k(), "Precision@K:%f" % rank_eval3.precision_at_k(), "Recall@K:%f" % rank_eval3.recall_at_k())

# COMMAND ----------

rank_eval4 = SparkRankingEvaluation(rating_true = test_df2, 
                                   rating_pred= predictions, 
                                   k = 10, col_user='asin1', col_item='asin2', 
                                   col_rating='review_cnts', col_prediction='prob', 
                                   relevancy_method='top_k', threshold=0.5)

print("Model: {}".format('GBDT(top k=10, test data co-review >=2)'), "Top K:%d" % rank_eval4.k, "MAP:%f" % rank_eval4.map_at_k(), "NDCG:%f" % rank_eval4.ndcg_at_k(), "Precision@K:%f" % rank_eval4.precision_at_k(), "Recall@K:%f" % rank_eval4.recall_at_k())

# COMMAND ----------



# COMMAND ----------


