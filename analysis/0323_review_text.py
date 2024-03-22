# Databricks notebook source
import pandas as pd
import json
import numpy as np
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import pyspark.pandas as ps

# COMMAND ----------

path = 'dbfs:/FileStore/amazon/data/All_Amazon_Review_Sample50000.json'

# COMMAND ----------

review = ps.read_json(path, lines=True)

# COMMAND ----------

display(review)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col

# SparkSession 생성
spark = SparkSession.builder \
    .appName("example") \
    .getOrCreate()

df = spark.read.json(path)

# 리뷰 텍스트 토큰화
words = df.select(explode(split(col("reviewText"), " ")).alias("word"))

# 워드 카운트 계산
word_count = words.groupBy("word").count()

word_count.show()

# COMMAND ----------

word_count_sort = words.groupBy("word").count().orderBy("count", ascending=False)

word_count_sort.show()

# COMMAND ----------

display(words)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. 소문자변환

# COMMAND ----------

from pyspark.sql.functions import lower

df = df.withColumn("reviewText", lower(df["reviewText"]))

# COMMAND ----------

display(df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. 태그값 제거 : \n

# COMMAND ----------

from pyspark.sql.functions import regexp_replace

# COMMAND ----------

df = df.withColumn("reviewText", regexp_replace("reviewText", "\n", ""))

# COMMAND ----------

words = df.select(explode(split(col("reviewText"), " ")).alias("word"))

word_count = words.groupBy("word").count()
word_count.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. 문장부호 및 특수문자 제거

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# 문장부호를 제거하는 함수 정의
def remove_punctuation(text):
    if text is None:  # None 값 처리
        return None
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    return ''.join(char for char in text if char not in punctuation)

# UDF 등록
remove_punctuation_udf = udf(remove_punctuation, StringType())

# reviewText 열에 적용하여 문장부호 제거
df = df.withColumn("cleanReviewText", remove_punctuation_udf("reviewText"))

# COMMAND ----------

words = df.select(explode(split(col("cleanReviewText"), " ")).alias("word"))

word_count = words.groupBy("word").count()
word_count.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. 불용어 제거
# MAGIC - ["i", "he", "she", "is", "am", "the", "a", "an", "are", "was", "were", "it", "that", "this", "these", "those","to"]

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, split
from pyspark.sql.types import StringType

# 불용어 처리 함수 정의
def remove_stopwords(text):
    if text is None:  # None 값인 경우 처리
        return None
    stopwords = ["i", "he", "she", "is", "am", "the", "a", "an", "are", "was", "were", "it", "that", "this", "these", "those", "to"]  # 불용어 리스트
    words = text.split()  # 공백을 기준으로 텍스트를 단어로 분할
    filtered_words = [word for word in words if word.lower() not in stopwords]  # 불용어가 아닌 단어만 선택
    if not filtered_words:  # 불용어만 있는 경우 처리
        return None
    return ' '.join(filtered_words)  # 필터링된 단어를 다시 공백으로 연결하여 반환

# UDF 등록
remove_stopwords_udf = udf(remove_stopwords, StringType())

# reviewText 열에 적용하여 불용어 제거
df = df.withColumn("cleanReviewText", remove_stopwords_udf("cleanReviewText"))

# 리뷰 텍스트 토큰화
words = df.select(explode(split(col("cleanReviewText"), " ")).alias("word"))

# 워드 카운트 계산 후 내림차순 정렬
word_count_sort = words.groupBy("word").count().orderBy("count", ascending=False)

# 결과 출력
word_count_sort.show()

# COMMAND ----------

df.filter(df.cleanReviewText.isNull()).show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. 토큰화: 텍스트를 단어 단위로 분할

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

# RegexTokenizer 객체 생성
tokenizer = RegexTokenizer(inputCol="cleanReviewText", outputCol="tokenized_text", pattern="\s+")

# 토큰화 수행
tokenized_df = tokenizer.transform(df)
tokenized_df.show(truncate=False)

# COMMAND ----------

display(tokenized_df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. 어간 추출 
# MAGIC - 단어의 접사 등을 제거하여 동일한 어간을 가진 단어를 동일한 형태로 표현
# MAGIC - "running", "runs", "ran"과 같은 단어들의 어간은 "run"

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. 워드 임베딩 : 토큰화한 단어에 실수를 부여하고, 벡터화하는 것

# COMMAND ----------


