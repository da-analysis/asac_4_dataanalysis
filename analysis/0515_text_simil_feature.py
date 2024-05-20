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
from pyspark.sql.functions import desc

# COMMAND ----------

df = spark.read.table("asac.text_sim_3")

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### q1, q2, kurtosis 유사도만 평균 (cosine_top3)

# COMMAND ----------

# 최종 코사인 유사도 평균 계산
df = df.withColumn("cosine_top3", (col("q1_cosine_similarity")                                 +col("q2_cosine_similarity")
+col("kurtosis_cosine_similarity")) / 3)

# COMMAND ----------

display(df)

# COMMAND ----------

name = "asac.text_sim_4"
df.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.text_sim_4

# COMMAND ----------


