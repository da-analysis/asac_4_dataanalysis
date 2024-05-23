#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# GBT 모델 학습 및 MLflow 저장

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import mlflow
import mlflow.spark

# Spark 세션 초기화
spark = SparkSession.builder.appName("modelComparision").getOrCreate()

# 데이터 로드 (여기서는 'df_filtered_balanced'를 로드하는 코드는 생략되어 있습니다.)
# 예시: df_filtered_balanced = spark.read.csv("path_to_file.csv", header=True, inferSchema=True)

# 목적 변수와 피처를 분리합니다.
target = "target"
features = [c for c in df_filtered_balanced.columns if c not in ['review_cnts', 'target', 'Year', 'Month', 'asin1', 'asin2']]

# VectorAssembler를 사용하여 피처 벡터를 생성합니다.
assembler = VectorAssembler(inputCols=features, outputCol="features")

# 데이터를 8:2 비율로 분할합니다.
train_data, test_data = df_filtered_balanced.randomSplit([0.8, 0.2], seed=42)

# 모델 초기화
gbt = GBTClassifier(featuresCol='features', labelCol=target)

# 모델 평가를 위한 evaluator 초기화
binary_evaluator = BinaryClassificationEvaluator(labelCol=target)
multi_evaluator = MulticlassClassificationEvaluator(labelCol=target, metricName="f1")

# 파이프라인 구성
pipeline = Pipeline(stages=[assembler, gbt])

# 모델 학습
model_fit = pipeline.fit(train_data)

# 예측
predictions = model_fit.transform(test_data)

# 성능 평가
auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
f1 = multi_evaluator.evaluate(predictions)
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})

# MLflow에 로깅
with mlflow.start_run():
    mlflow.spark.log_model(model_fit, "GBT_Model")  # Log the model
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)

