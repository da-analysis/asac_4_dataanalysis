# Databricks notebook source
# MAGIC %md
# MAGIC ## 감성 분석 (모든 데이터 합쳐서)

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

# 테이블 읽기
cell_meta = spark.read.table("asac.meta_cell_phones_and_accessories_new_price2")
sport_meta = spark.read.table("asac.sports_and_outdoors_fin_v2")
cell_review = spark.read.table("asac.review_cellphone_accessories_final")
sport_review = spark.read.table("asac.reivew_sports_outdoor_final")


# pyspark pandas DataFrame으로 변경
cell_meta = ps.DataFrame(cell_meta)
sport_meta = ps.DataFrame(sport_meta)
cell_review = ps.DataFrame(cell_review)
sport_review = ps.DataFrame(sport_review)

# COMMAND ----------

display(cell_review)

# COMMAND ----------

# reviewText, asin, reviewerID 만 뽑아서 테이블 합치기
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

cell_review_text = spark.read.table("asac.review_cellphone_accessories_final").select("reviewText","asin","reviewerID","overall","reviewTime")
sport_review_text = spark.read.table("asac.reivew_sports_outdoor_final").select("reviewText","asin","reviewerID","overall","reviewTime")

review_text = cell_review_text.union(sport_review_text)

review_text.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. vivekn 모델 감성 분석
# MAGIC - https://sparknlp.org/2021/11/22/sentiment_vivekn_en.html

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

# COMMAND ----------

review_text.createOrReplaceTempView("review_text")

# COMMAND ----------

display(review_text)

# COMMAND ----------

document = DocumentAssembler() \
.setInputCol("reviewText") \
.setOutputCol("document")

token = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

normalizer = Normalizer() \
.setInputCols(["token"]) \
.setOutputCol("normal")

vivekn =  ViveknSentimentModel.pretrained() \
.setInputCols(["document", "normal"]) \
.setOutputCol("result_sentiment")

finisher = Finisher() \
.setInputCols(["result_sentiment"]) \
.setOutputCols("final_sentiment")

pipeline = Pipeline().setStages([document, token, normalizer, vivekn, finisher])

# COMMAND ----------

pipelineModel = pipeline.fit(review_text)
result = pipelineModel.transform(review_text)

# COMMAND ----------

display(result)

# COMMAND ----------

result = result.withColumn("final_sentiment", explode('final_sentiment'))
display(result)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. 허깅페이스 transformer 활용
# MAGIC - 스타, 점수 추출해서 피쳐로 만들기
# MAGIC - 1 star는 매우 부정적, 3stars는 중립, 5stars는 매우 긍정적
# MAGIC - score는 해당감정에 속할 확률
# MAGIC
# MAGIC - 작업이 오래걸리면 mulitprocessing 라이브러리 참고해서 코드 수정해서 돌려보기 (or joblib)
# MAGIC - 입력데이터 조건이 있음... 512....이거 잘라서 할지??
# MAGIC - show는 되는데, display 안되는 경우

# COMMAND ----------

!pip install --upgrade pip

# COMMAND ----------

!pip install huggingface_hub

# COMMAND ----------

!pip install langchain_community
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# COMMAND ----------

!pip install torch torchvision torchaudio

# COMMAND ----------

# MAGIC %pip install torch torchvision

# COMMAND ----------

# MAGIC %pip install transformers
# MAGIC %pip install langchain_community
# MAGIC from langchain_community.embeddings import HuggingFaceHubEmbeddings

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 감정 분석을 위한 사전 훈련된 모델 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 파이프라인 설정 (감정 분석)
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 문서 예시
documents = [
    "I love Hugging Face!", 
    "I hate Hugging Face...",
    "Hugging Face is just okay."
]

# 문서 감정 분석
for doc in documents:
    result = sentiment_analysis_pipeline(doc)
    print(f"Document: {doc}")
    print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")
    print()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()


# PyTorch 및 Transformers 라이브러리 import
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 감정 분석을 위한 사전 훈련된 모델 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 파이프라인 설정 (감정 분석)
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# UDF 정의
def analyze_sentiment(text):
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label}, Score: {score}"

# UDF 등록
analyze_sentiment_udf = udf(analyze_sentiment, StringType())

# review_text_df에 UDF 적용
result_df = review_text.limit(10).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

# 결과 출력
result_df.show(truncate=False)


# COMMAND ----------

display(result_df)

# COMMAND ----------

# UDF 정의
def analyze_sentiment(text):
    #text = text[:512]  #  maximum 512 tokens
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# UDF 등록 및 스키마 정의
analyze_sentiment_udf = udf(analyze_sentiment, StructType([
    StructField("Sentiment", StringType(), True),
    StructField("Score", FloatType(), True)
]))

# review_text_df에 UDF 적용
result_df = review_text.limit(100).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))


# COMMAND ----------

display(result_df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()


# PyTorch 및 Transformers 라이브러리 import
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 감정 분석을 위한 사전 훈련된 모델 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 파이프라인 설정 (감정 분석)
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# UDF 정의
def analyze_sentiment(text):
    text = text[:512]
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label}, Score: {score}"

# UDF 등록
analyze_sentiment_udf = udf(analyze_sentiment, StringType())

# review_text_df에 UDF 적용
result_df = review_text.limit(1000).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

# 결과 출력
result_df.show(truncate=False)


# COMMAND ----------

display(result_df)

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import udf

def extract_star(label):
    try:
        star = int(label.split()[0])
        return star
    except ValueError:
        return None


# UDF 등록
extract_star_udf = udf(extract_star, IntegerType())

result_df = result_df.withColumn("star", extract_star_udf("sentiment_analysis"))

result_df = result_df.withColumn("score", result_df["sentiment_analysis"].substr(-7, 7).cast(FloatType()))

# 결과 출력
result_df.show(truncate=False)

# COMMAND ----------

display(result_df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. 긍부정으로 나오는 것
# MAGIC - 여기서 score는 해당 감정에 속할 확률을 나타냄

# COMMAND ----------

!pip install transformers

# COMMAND ----------

from transformers import pipeline

# COMMAND ----------

!pip install torch torchvision torchaudio

# COMMAND ----------

# MAGIC %pip install torch torchvision

# COMMAND ----------

from transformers import pipeline

# build pipeline
classifier = pipeline("sentiment-analysis")

#inference
classifier("we are very happy to show you the 🤗 Transformers library.")

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType, StructType, StructField

classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    text = text[:512]  #  maximum 512 tokens
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return label, score

analyze_sentiment_udf = udf(analyze_sentiment, StructType([
    StructField("label", StringType(), True),
    StructField("score", FloatType(), True)
]))

result_df = review_text.withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

result_df = result_df.withColumn("label", result_df["sentiment_analysis"].getField("label")) \
                     .withColumn("score", result_df["sentiment_analysis"].getField("score")) \
                     .drop("sentiment_analysis")

result_df.show(truncate=False)


# COMMAND ----------

display(result_df.head(100))

# COMMAND ----------


