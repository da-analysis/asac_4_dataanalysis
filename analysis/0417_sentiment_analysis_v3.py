# Databricks notebook source
# MAGIC %md
# MAGIC ## 감성 분석 (모든 데이터 합쳐서)

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

review_text = spark.read.table("asac.senti_review_text")
review_text = ps.DataFrame(review_text)

# COMMAND ----------

display(review_text)

# COMMAND ----------



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

num_partitions = 4
name = "asac.senti_vivekn_fin"

result_vivekn.repartition(num_partitions).write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_vivekn_fin

# COMMAND ----------

vivekn_df = spark.read.table("asac.senti_vivekn_fin")
vivekn_df = ps.DataFrame(vivekn_df)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. 허깅페이스 transformer 활용 => sparkNlp에 있는거 우선 먼저 활용
# MAGIC - https://sparknlp.org/2021/11/03/bert_sequence_classifier_multilingual_sentiment_xx.html 
# MAGIC - https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment?text=I+like+you.+I+love+you 
# MAGIC - 스타, 점수 추출해서 피쳐로 만들기
# MAGIC - 1 star는 매우 부정적, 3stars는 중립, 5stars는 매우 긍정적
# MAGIC - score는 해당감정에 속할 확률
# MAGIC
# MAGIC - 작업이 오래걸리면 mulitprocessing 라이브러리 참고해서 코드 수정해서 돌려보기 (or joblib)
# MAGIC - 입력데이터 조건이 있음... 512....이거 잘라서 할지??
# MAGIC - show는 되는데, display 안되는 경우

# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

# COMMAND ----------

name3 = "asac.senti_review_text"
review_text_df = spark.read.table(name3)

# COMMAND ----------

review_text_df_10 = review_text_df.limit(10)

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

display(result)

# COMMAND ----------



# COMMAND ----------

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

# COMMAND ----------

name3 = "asac.senti_review_text"
review_text_df = spark.read.table(name3)

# COMMAND ----------

review_text_df_10 = review_text_df.limit(10)

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

display(result)

# COMMAND ----------

name = "asac.senti_trans_1000"
result_trans.limit(1000).write.saveAsTable(name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 멀티프로세싱 100개씩 10번 해보기

# COMMAND ----------

name3 = "asac.senti_review_text"
review_text_df = spark.read.table(name3)

# COMMAND ----------

mport sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from multiprocessing import Pool


# 파이프라인 설정
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

result = pipeline.fit(review_text_df).transform(review_text_df)




# COMMAND ----------

import multiprocessing
multiprocessing.cpu_count()

# COMMAND ----------

!pip install ray
!pip install torch
!pip install transformers
!pip install scikit-learn
!pip install psutil

# COMMAND ----------

import ray
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml import Pipeline
from sparknlp.annotator import DocumentAssembler, Tokenizer, BertForSequenceClassification

ray.shutdown()

# Initialize Ray
ray.init()

# Define a function to apply the pipeline on a chunk of data
@ray.remote
def process_chunk(chunk):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SparkNLP Pipeline") \
        .getOrCreate()
    
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

    pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

    result = pipeline.fit(chunk).transform(chunk)
    
    # Stop Spark session
    spark.stop()
    
    return result

if __name__ == '__main__':
    # Assume review_text_df is the DataFrame containing the data

    # Split the DataFrame into chunks (e.g., 100 rows per chunk)
    chunks = [chunk for chunk in review_text_df.randomSplit([1.0]*100)]

    # Apply the pipeline on each chunk using multiple CPU cores in parallel
    processed_chunks = ray.get([process_chunk.remote(chunk) for chunk in chunks])

    # Concatenate the processed chunks back into a single DataFrame
    result_trans = processed_chunks[0]
    for chunk in processed_chunks[1:]:
        result_trans = result_trans.union(chunk)

    # Save the result DataFrame
    name = "asac.senti_trans_temp"
    result_trans.write.saveAsTable(name)

    # Shutdown Ray
    ray.shutdown()


# COMMAND ----------

#######################################
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from multiprocessing import Pool


# 파이프라인 설정
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


pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

spark = SparkSession.builder \
    .appName("Multiprocessing Example") \
    .getOrCreate()

# 병렬 처리 함수 정의
def process_chunk(chunk):
    result_chunk = pipeline.fit(chunk).transform(chunk)
    return result_chunk

if __name__ == '__main__':
    # review_text_df는 처리할 데이터프레임
    chunks = [review_text_df.limit(10) for _ in range(10)]  # 10개씩 10번 처리하도록 설정

    # multiprocessing을 사용하여 병렬로 처리
    with Pool() as pool:
        processed_chunks = pool.map(process_chunk, chunks)

    for i, result_chunk in enumerate(processed_chunks):
        result_chunk.write.saveAsTable(f"asac.senti_trans_multi_{i}", mode="overwrite")

    spark.stop()



# COMMAND ----------

review_text_df_10 = review_text_df.limit(10)

# COMMAND ----------

result = pipeline.fit(review_text_df_10).transform(review_text_df_10)

# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC 판다스 데이터프레임으로 바꿔서 해보기

# COMMAND ----------

import pandas as pd
name3 = "asac.senti_review_text"
review_text_df = spark.read.table(name3)

# Spark 데이터프레임을 판다스 데이터프레임으로 변환
review_text_pd_df = review_text_df.toPandas()


# COMMAND ----------

review_text_pd_df.info()

# COMMAND ----------

review_text_df

# COMMAND ----------

name = "asac.senti_trans_1000"
result_trans.limit(1000).write.saveAsTable(name)

# COMMAND ----------

review_text_pd_df_100 = review_text_pd_df[:100]

# COMMAND ----------

review_text_pd_df_100.head(5)

# COMMAND ----------

from pyspark.sql import SparkSession
from joblib import Parallel, delayed
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from sparknlp.document import DocumentAssembler
from sparknlp.base import BertForSequenceClassification

# 병렬 처리를 수행할 Python 함수 정의
def process_row(row):
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

    pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])
    row = pipeline.fit(row).transform(row)

    return row

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

# Apply process_row function to each row using map
processed_data = review_text_pd_df_100.rdd.map(process_row)

# Convert RDD of processed rows to DataFrame
processed_df = processed_data.toDF()


# COMMAND ----------


# 처리 결과 저장
processed_df.write.saveAsTable("processed_data_table")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install --upgrade pip
# MAGIC !pip install --upgrade pip
# MAGIC !pip install huggingface_hub
# MAGIC %pip install spark-nlp
# MAGIC %pip install torch torchvision
# MAGIC !pip install torch torchvision torchaudio
# MAGIC %pip install transformers
# MAGIC %pip install langchain_community
# MAGIC from langchain_community.embeddings import HuggingFaceHubEmbeddings

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

review_text = spark.read.table("asac.senti_review_text")

result_trans= review_text.withColumn("sentiment_analysis", analyze_sentiment_udf("reviewText"))

# COMMAND ----------

result_trans.show(10,truncate=False)

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

name = "asac.senti_trans_10"
result_trans.limit(10).write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_trans_10

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table asac.senti_trans_10

# COMMAND ----------

name = "asac.senti_trans_100"
result_trans.limit(100).write.saveAsTable(name)

# COMMAND ----------

total_rows = result_trans.count()
total_rows

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

name = "asac.senti_trans_while20"
total_rows = result_trans.limit(10000).count()
batch_size = 100
start_row = 0

while start_row < total_rows:
    # 현재 저장된 행의 범위에서 다음 100개의 데이터 선택
    batch_df = result_trans.limit(10000).withColumn("row_number", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))) \
                           .filter(col("row_number").between(start_row + 1, start_row + batch_size))
    
    # 이전에 저장된 데이터를 건너뛰고 다음 100개 데이터를 선택
    if start_row > 0:
        # 새로운 배치 데이터프레임을 기존 데이터프레임에 추가
        batch_df = batch_df.select(*result_trans.columns)
    
    # 선택된 데이터를 테이블에 추가로 저장
    batch_df.drop("row_number").write.mode("append").saveAsTable(name)
    
    # 다음 반복을 위해 시작 행을 업데이트
    start_row += batch_size 


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_trans_while20

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
# MAGIC select overall, final_sentiment from asac.senti_vivekn_fin

# COMMAND ----------

# MAGIC %sql
# MAGIC select overall, final_sentiment from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="positive"

# COMMAND ----------

# MAGIC %sql
# MAGIC select overall, final_sentiment from asac.senti_vivekn_fin
# MAGIC where final_sentiment=="negative"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT overall, final_sentiment  FROM asac.senti_vivekn_fin
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
# MAGIC select mean(overall), min(overall), max(overall), median(overall),count(overall) from asac.senti_vivekn_fin
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


