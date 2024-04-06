# Databricks notebook source
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

# reviewText만 뽑아서 테이블 합치기
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

cell_review_text = spark.read.table("asac.review_cellphone_accessories_final").select("reviewText")
sport_review_text = spark.read.table("asac.reivew_sports_outdoor_final").select("reviewText")

merged_data = cell_review_text.union(sport_review_text)

merged_data.show(truncate=False)

# COMMAND ----------

cell_meta.info()

# COMMAND ----------

sport_meta.info()

# COMMAND ----------

cell_review.info()

# COMMAND ----------

sport_review.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### drop column 해보기

# COMMAND ----------

# MAGIC %sql
# MAGIC alter table asac.sports_and_outdoors_fin_v2
# MAGIC drop column new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE asac.sports_and_outdoors_fin_v2 SET TBLPROPERTIES (
# MAGIC    'delta.columnMapping.mode' = 'name',
# MAGIC    'delta.minReaderVersion' = '2',
# MAGIC    'delta.minWriterVersion' = '5')

# COMMAND ----------

# MAGIC %sql
# MAGIC alter table asac.sports_and_outdoors_fin_v2
# MAGIC drop column new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE asac.sports_and_outdoors_fin_v2 
# MAGIC RENAME COLUMN new_price3 TO new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

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

# reviewText만 뽑아서 테이블 합치기
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

cell_review_text = spark.read.table("asac.review_cellphone_accessories_final").select("reviewText")
sport_review_text = spark.read.table("asac.reivew_sports_outdoor_final").select("reviewText")

merged_data = cell_review_text.union(sport_review_text)

merged_data.show(truncate=False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Ngram 및 텍스트 전처리 리서치 및 진행
# MAGIC ### 워드 클라우드 나타내기
# MAGIC
# MAGIC ##### N-gram(엔그램) 은 Bag of Words , TF-IDF 와 같이 횟수를 사용하여 단어를 벡터로 표현(Count-based representation) 하는 방법입니다. 이전 두 방법을 사용할 때에는 한 개의 단어만을 분석 대상으로 삼았지만 N-gram은 한 단어 이상의 단어 시퀀스를 분석 대상으로 삼습니다. N-gram 앞에 N에 따라서 단어 시퀀스를 몇 개의 단어로 구성할 지를 결정하게 됩니다. N=1,2,3인 경우를 각각 Uni-gram, Bi-gram, Tri-gram 이라 부르며 N>=4 일 때는 그냥 4-gram, 5-gram 의 방식으로 부릅니다. 
# MAGIC

# COMMAND ----------

cell_review.info()

# COMMAND ----------

sport_review.info()

# COMMAND ----------

display(merged_data.head(5))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 태그 제거

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace

# SparkSession 생성
spark = SparkSession.builder \
    .appName("RemoveTagsFromText") \
    .getOrCreate()

# 태그 제거
merged_data = merged_data.withColumn("no_tag", regexp_replace("reviewText", "[\\n\\t\/<>]", ""))

display(merged_data.head(5))


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 문장부호 제거

# COMMAND ----------

# SparkSession 생성
spark = SparkSession.builder \
    .appName("RemovePunctuationFromColumn") \
    .getOrCreate()

# 문장부호 제거
merged_data = merged_data.withColumn("no_func", regexp_replace("no_tag", r"[!?,.\(\)\[\]\{\}_:;#@]", " "))

display(merged_data.head(20))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 소문자 통일, 공백 분리(tokenizer)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

tokenizer = Tokenizer(inputCol="no_func", outputCol="token")
merged_data = tokenizer.transform(merged_data)

display(merged_data.head(15))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 불용어 제거

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = list(stopwords.words('english'))

# COMMAND ----------

print(len(stop_words), stop_words[:10])

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(stopWords=stop_words)
remover.setInputCol("token")
remover.setOutputCol("stopword")
merged_data = remover.transform(merged_data)
display(merged_data.head(15))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC -공백제거

# COMMAND ----------

blank = [""]

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(stopWords=blank)
remover.setInputCol("stopword")
remover.setOutputCol("no_blank")
merged_data = remover.transform(merged_data)
display(merged_data.head(15))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 어간 추출

# COMMAND ----------

!pip install spark-nlp
import sparknlp
from sparknlp.annotator import Stemmer, LemmatizerModel
from sparknlp.base import DocumentAssembler, Pipeline
spark = sparknlp.start()

# COMMAND ----------

!pip install BeautifulSoup4

# COMMAND ----------

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# COMMAND ----------

!pip install spark-nlp


# COMMAND ----------

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
spark = sparknlp.start()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - ngram 진행 ngram(2), ngram(3)

# COMMAND ----------

from pyspark.ml.feature import NGram
from pyspark.ml import Pipeline

ngram = NGram(n=2, inputCol="no_blank", outputCol="2grams")
pipeline = Pipeline(stages=[ngram])
model = pipeline.fit(merged_data)
review_ngram = model.transform(merged_data)

display(review_ngram.head(15))

# COMMAND ----------

from pyspark.ml.feature import NGram
from pyspark.ml import Pipeline

ngram = NGram(n=3, inputCol="no_blank", outputCol="3grams")
pipeline = Pipeline(stages=[ngram])
model = pipeline.fit(review_ngram)
review_ngram = model.transform(review_ngram)

display(review_ngram.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC - 워드클라우드

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

display(review_ngram.head(15))

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import explode, concat_ws, collect_list

# 2grams 열을 explode하여 각 단어를 개별 행으로 변환
exploded_df = review_ngram.withColumn("2grams_word", explode("2grams"))

# COMMAND ----------

display(exploded_df)

# COMMAND ----------

display(review_ngram.head(10))

# COMMAND ----------

word_df = exploded_df.withColumn("combined_words", concat_ws(" ", "2grams_word"))
word_df = word_df.fillna("", subset=["combined_words"])

# COMMAND ----------

display(word_df.head(100))

# COMMAND ----------

all_words = " ".join(word_df.select("combined_words").rdd.flatMap(lambda x: x).collect())

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install wordcloud

# COMMAND ----------



# COMMAND ----------

# MAGIC %sh
# MAGIC pip install wordcloud

# COMMAND ----------

!sudo apt install fontconfig

# COMMAND ----------

!sudo fc-cache -f -v

# COMMAND ----------

!fc-list

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /etc/issue

# COMMAND ----------

!pip install wordcloud

# COMMAND ----------

from wordcloud import WordCloud
from collections import Counter

word_ls = ["aa dd","ss d","aa dd","ew sq"]
word_cloud_dict = Counter(word_ls) 

# TrueType 폰트 경로를 지정하여 WordCloud 생성
wordcloud = WordCloud(font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf").generate_from_frequencies(word_cloud_dict) 

# 워드클라우드 시각화
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# COMMAND ----------

font_path = "C:\\Windows\\Fonts\\ariblk.ttf"
# 또는
font_path = r"C:\Windows\Fonts\ariblk.ttf"


# COMMAND ----------

display(word_cloud_dict)

# COMMAND ----------

from collections import Counter

# Convert the Spark SQL Column to a Python list
word_list = word_df.toPandas()['2grams_word'].tolist()

# Count the occurrences of each word
word_count_dict = Counter(word_list)

wordcloud = WordCloud().generate_from_frequencies(word_count_dict)

# COMMAND ----------

from collections import Counter

# 2grams_word 열의 값을 리스트로 변환
word_list = exploded_df.select("2grams_word").rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

display(exploded_df.head(10))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 감정 분석

# COMMAND ----------

!pip install spark-nlp
!pip install pyspark

# COMMAND ----------

import sparknlp
# Start Spark Session
spark = sparknlp.start()

# COMMAND ----------

# Import the required modules and classes
from sparknlp.base import DocumentAssembler, Pipeline, Finisher
from sparknlp.annotator import (
    SentenceDetector,
    Tokenizer,
    Lemmatizer,
    SentimentDetector
)
import pyspark.sql.functions as F
# Step 1: Transforms raw texts to `document` annotation
document_assembler = (
    DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
)
# Step 2: Sentence Detection
sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
# Step 3: Tokenization
tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
# Step 4: Lemmatization
lemmatizer= Lemmatizer().setInputCols("token").setOutputCol("lemma")
                        .setDictionary("lemmas_small.txt", key_delimiter="->", value_delimiter="\t")
# Step 5: Sentiment Detection
sentiment_detector= (
    SentimentDetector()
    .setInputCols(["lemma", "sentence"])
    .setOutputCol("sentiment_score")
    .setDictionary("default-sentiment-dict.txt", ",")
)
# Step 6: Finisher
finisher= (
    Finisher()
    .setInputCols(["sentiment_score"]).setOutputCols("sentiment")
)
# Define the pipeline
pipeline = Pipeline(
    stages=[
        document_assembler,
        sentence_detector, 
        tokenizer, 
        lemmatizer, 
        sentiment_detector, 
        finisher
    ]
)

# COMMAND ----------

# Create a spark Data Frame with an example sentence
data = spark.createDataFrame(
    [
        [
            "The restaurant staff is really nice"
        ]
    ]
).toDF("text") # use the column name `text` defined in the pipeline as input
# Fit-transform to get predictions
result = pipeline.fit(data).transform(data).show(truncate = 50)

# COMMAND ----------

# Create a spark Data Frame with an example sentence
data = spark.createDataFrame(
    [
        [
            "I recommend others to avoid because it is too expensive"
        ]
    ]
).toDF("text") # use the column name `text` defined in the pipeline as input
# Fit-transform to get predictions
result = pipeline.fit(data).transform(data).show(truncate = 50)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 허깅페이스 transformers
# MAGIC

# COMMAND ----------

pip install transformers

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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



# COMMAND ----------

!pip install huggingface_hub

# COMMAND ----------

!pip install --upgrade pip

# COMMAND ----------

!pip install langchain_community
from langchain_community.embeddings import HuggingFaceHubEmbeddings


# COMMAND ----------




# COMMAND ----------

query_result = embeddings.embed_query(text)

# COMMAND ----------

query_result[:3]

# COMMAND ----------

embeddings = HuggingFaceHubEmbeddings()
text = (
    "hello my name is jyp. i'm tired."  
)

# COMMAND ----------

query_result = embeddings.embed_query(text)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 허깅페이스 GPT2

# COMMAND ----------

!pip install transformers
!pip install torch torchvision torchaudio

# COMMAND ----------

from transformers import AutoTokenizer, GPT2Model
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# COMMAND ----------

# 임베딩할 대상 문장
sequence = "Hello, my dog is cute"

# 토큰화
inputs = tokenizer(sequence, return_tensors="pt")

# 임베딩
outputs = model(**inputs)

# 임베딩 결과 확인
print("embedding size:   ", outputs.last_hidden_state.size())
print("embedding vector: ", outputs.last_hidden_state)

# COMMAND ----------

# 두 개의 문장 확인
# 패딩 토큰 지정
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token

# 두 개의 문장을 포함하는 배열 (문장의 길이가 약간 다르다)
sequence = ["Hello, my dog is cute isnt it?", "Hello, my dog is super cute isnt it?"]

# 토큰화
inputs = tokenizer(sequence, return_tensors="pt", 
	padding=True, truncation=True)

# 임베딩
outputs = model(**inputs)

# 임베딩 결과 확인
print("embedding size:   ", outputs.last_hidden_state.size())
print("embedding vector: ", outputs.last_hidden_state)

# COMMAND ----------


