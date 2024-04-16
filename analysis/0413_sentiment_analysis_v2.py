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

result_vivekn = result.withColumn("final_sentiment", explode('final_sentiment'))
display(result_vivekn)

# COMMAND ----------

# MAGIC %md
# MAGIC -> 텍스트 있는데도, 결과가 na가 나오기도함

# COMMAND ----------

# MAGIC %md
# MAGIC 델타테이블로 저장하기

# COMMAND ----------

name = "asac.senti_vivekn_limit10"
result_vivekn.limit(10).write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_vivekn_limit10

# COMMAND ----------

# MAGIC %md
# MAGIC 너무 오래걸림... parquet 파일로 저장하고 파티션 나눠서 저장하기

# COMMAND ----------

#path1 = "dbfs:/FileStore/amazon/review/"
#result_vivekn.repartition(4).write.parquet('%s/to_parquet/result_vivekn.parquet' % path1, compression='zstd')

# path = "dbfs:/FileStore/amazon/review/to_parquet/result_vivekn.parquet"
# df_vivekn = spark.read.parquet(path)

# COMMAND ----------

num_partitions = 4
name = "asac.senti_vivekn11"

result_vivekn.limit(11).repartition(num_partitions).write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_vivekn11

# COMMAND ----------

display(result_vivekn)

# COMMAND ----------

num_partitions = 4
name = "asac.senti_vivekn_fin"

result_vivekn.repartition(num_partitions).write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

name = "asac.senti_vivekn_limit300"
result_vivekn.limit(300).write.saveAsTable(name)

# COMMAND ----------

name = "asac.senti_vivekn_limit5000"
result_vivekn.limit(5000).write.saveAsTable(name)

# COMMAND ----------

name = "asac.senti_vivekn_limit50000"
result_vivekn.limit(50000).write.saveAsTable(name)

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

df = review_text.limit(100)

# COMMAND ----------

num_partitions = 4
name3 = "asac.review_text_senti"

review_text.repartition(num_partitions).write.saveAsTable(name3, mode="overwrite")

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

# COMMAND ----------

name3 = "asac.review_text_senti"
review_text_df = spark.read.table(name3)

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol('reviewText') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_multilingual_sentiment', 'xx') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(False) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

# COMMAND ----------


result = pipeline.fit(review_text_df).transform(review_text_df)

# COMMAND ----------

#num_partitions = 4
name2 = "asac.senti_trans"

result.write.saveAsTable(name2, mode="overwrite")

# COMMAND ----------

display(result)

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

# reviewText, asin, reviewerID 만 뽑아서 테이블 합치기
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

cell_review_text = spark.read.table("asac.review_cellphone_accessories_final").select("reviewText","asin","reviewerID","overall","reviewTime")
sport_review_text = spark.read.table("asac.reivew_sports_outdoor_final").select("reviewText","asin","reviewerID","overall","reviewTime")

review_text = cell_review_text.union(sport_review_text)

review_text.show(truncate=False)
review_text.createOrReplaceTempView("review_text")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .getOrCreate()

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 감정 분석을 위한 사전 훈련된 모델 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label}, Score: {score}"

# UDF 등록
analyze_sentiment_udf = udf(analyze_sentiment, StringType())

# review_text_df에 UDF 적용(10개만 해보기)
result_df = review_text.limit(10).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

result_df.show(truncate=False)

# COMMAND ----------

display(result_df)

# COMMAND ----------

def analyze_sentiment(text):
    #text = text[:512]  #  maximum 512 tokens
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

analyze_sentiment_udf = udf(analyze_sentiment, StructType([
    StructField("Sentiment", StringType(), True),
    StructField("Score", FloatType(), True)
]))
# 10개만 확인해보기
result_df = review_text.limit(10).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))
display(result_df)

# COMMAND ----------



# COMMAND ----------

def analyze_sentiment(text):
    #text = text[:512]  #  maximum 512 tokens
    result = sentiment_analysis_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

analyze_sentiment_udf = udf(analyze_sentiment, StructType([
    StructField("Sentiment", StringType(), True),
    StructField("Score", FloatType(), True)
]))
# 100개만 확인해보기
result_df = review_text.limit(100).withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

result_df.show(100)


# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC - 토큰이 512개가 넘는 리뷰 몇개 있는지 확인하기

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 감정 분석을 위한 사전 훈련된 모델 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# UDF 정의: 토큰화하고 토큰의 개수를 반환하는 함수
def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    # 토큰 개수
    return len(tokens)

count_tokens_udf = udf(count_tokens, IntegerType())

review_text_token = review_text_token.withColumn("token_cnt", count_tokens_udf(col("reviewText")))

review_text_token.show(230) #240은 안됨

# COMMAND ----------

display(review_text.head(250))

# COMMAND ----------

# MAGIC %md
# MAGIC - 238행에 null값이 있어서 그런거 같음.... 왜 널값...?
# MAGIC - 그럼 null값인 행을 삭제하고 다시 해보겠

# COMMAND ----------

review_text_nona = review_text.na.drop(subset=["reviewText"])

# COMMAND ----------

# MAGIC %md
# MAGIC - transformer 토크나이저 활용

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType

# UDF 정의: 텍스트를 토큰화하고 토큰의 개수를 반환하는 함수
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_text(text):
    # 텍스트를 토큰화
    tokens = tokenizer.tokenize(text)
    return tokens

tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

review_text_token = review_text_nona.withColumn("tokens", tokenize_text_udf(col("reviewText")))


from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

# UDF 정의: 토큰화하고 토큰의 개수를 반환하는 함수
def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    # 토큰 개수
    return len(tokens)

count_tokens_udf = udf(count_tokens, IntegerType())

review_text_token = review_text_token.withColumn("token_cnt", count_tokens_udf(col("reviewText")))

review_text_token.show(250) 

# COMMAND ----------

# 512를 초과 개수 
count_over_512 = review_text_token.filter(review_text_token["token_cnt"] > 512).count()
print(count_over_512)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 그럼 기본토크나이저와 transformer에서의 토크나이저에 차이가 있는지 확인
# MAGIC #### 없다면, 기본 토그타이저에서 512넘는 행을 삭제후 다시 돌려보기

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType
from sparknlp.annotator import Tokenizer
from sparknlp.base import DocumentAssembler

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# Tokenizer를 사용하여 문서를 토큰화
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

review_text_nona = review_text.na.drop(subset=["reviewText"])

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[documentAssembler, tokenizer])
pipeline_model = pipeline.fit(review_text_nona)
tokenized_data = pipeline_model.transform(review_text_nona)

from pyspark.sql.functions import size

# 토큰의 개수를 세는 UDF 정의
count_tokens_udf = udf(lambda tokens: len(tokens), IntegerType())
tokenized_data = tokenized_data.withColumn("token_cnt", count_tokens_udf(col("token.result")))

tokenized_data.show(100)

# COMMAND ----------

# transformer 토크나이저
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, ArrayType

text_column = review_text_nona.limit(100)


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# UDF 정의: 텍스트를 토큰화하고 토큰의 개수를 반환하는 함수
def tokenize_text(text):
    # 텍스트를 토큰화
    tokens = tokenizer.tokenize(text)
    return tokens

tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

review_text_token = text_column.withColumn("tokens", tokenize_text_udf(col("reviewText")))


# UDF 정의: 토큰화하고 토큰의 개수를 반환하는 함수
def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    # 토큰 개수
    return len(tokens)

count_tokens_udf = udf(count_tokens, IntegerType())

review_text_token = review_text_token.withColumn("token_cnt", count_tokens_udf(col("reviewText")))

review_text_token.show(100) 

# COMMAND ----------

# MAGIC %md
# MAGIC - 둘의 토큰화 방식이 다름
# MAGIC - 우선 그럼 transformer 모델 활용할 때는, 512넘는 행에 대해서는 삭제하기 보다는 512까지만 text를 활용하는게 더 나을것 같음
# MAGIC =>그럼 현재까지 vivekn과의 차이점은 리뷰텍스트가 na인 행을 삭제한 것과, 텍스트 512까지만 한 것으로 진행한다는 점!

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


analyze_sentiment_udf = udf(analyze_sentiment, StringType())

result_trans= review_text_nona.withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

# COMMAND ----------

result_trans.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC - star와 score열 따로 분리해서 가져오기

# COMMAND ----------

from pyspark.sql.functions import regexp_extract

# Sentiment 열 생성
result_trans = result_trans.withColumn("sentiment", regexp_extract("sentiment_analysis", r"(\d+) star", 1).cast(IntegerType()))


# Score 열 생성
result_trans = result_trans.withColumn("score", regexp_extract("sentiment_analysis", r"Score: ([0-9.]+)", 1).cast(FloatType()))

result_trans.show(100)

# COMMAND ----------

result_trans.show(300)

# COMMAND ----------

# MAGIC %md
# MAGIC 델타테이블로 저장하기

# COMMAND ----------

name = "asac.senti_trans10"
result_trans.limit(10).write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_trans10

# COMMAND ----------



# COMMAND ----------

num_partitions = 4
name2 = "asac.senti_trans"

result_trans.repartition(num_partitions).write.saveAsTable(name2, mode="overwrite")

# COMMAND ----------

#name = "asac.senti_trans"
#result_trans.write.saveAsTable(name)

# COMMAND ----------

name = "asac.senti_trans_100"
result_trans.limit(100).write.saveAsTable(name)

# COMMAND ----------

name = "asac.senti_trans_1000"
result_trans.limit(1000).write.saveAsTable(name)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### vivekn 모델, transformer 모델 탐색하기
# MAGIC - 긍부정결과 vs 리뷰 평점
# MAGIC - 어떤 분포를 보이는 지 확인하기

# COMMAND ----------

# MAGIC %sql
# MAGIC select overall from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="positive"

# COMMAND ----------

# MAGIC %sql
# MAGIC select overall from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="negative"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT overall  FROM asac.senti_vivekn_fin
# MAGIC WHERE final_sentiment !="positive" and final_sentiment !="negative"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT overall, final_sentiment  FROM asac.senti_vivekn_fin

# COMMAND ----------

# MAGIC %sql
# MAGIC select mean(overall), min(overall), max(overall), median(overall) from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="positive"
# MAGIC group by final_sentiment
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select mean(overall), min(overall), max(overall), median(overall) from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="negative"
# MAGIC group by final_sentiment

# COMMAND ----------

# MAGIC %sql
# MAGIC select mean(overall), min(overall), max(overall), median(overall) from asac.senti_vivekn_fin
# MAGIC group by final_sentiment
# MAGIC -- 1이 positive
# MAGIC -- 2가 negative
# MAGIC -- 3이 na

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. 긍부정으로 나오는 것(그냥 추가로 해본것, 얘는 더 안하기)
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


