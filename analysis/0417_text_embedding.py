# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text

# COMMAND ----------

# MAGIC %md
# MAGIC ### Doc2Vec
# MAGIC -> 벡터 개수 지정 가능
# MAGIC

# COMMAND ----------

# MAGIC %pip install gensim

# COMMAND ----------

import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# 예시 문서
documents = ["This is the first document.",
             "This document is the second document.",
             "And this is the third one.",
             "Is this the first document?"]

# 문서를 토큰화하고 TaggedDocument 형식으로 변환
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents)]

# Doc2Vec 모델 학습
model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4, epochs=100)

document_embeddings = [model.dv[str(i)] for i in range(len(documents))]
print(document_embeddings)


# COMMAND ----------

import numpy as np

# 문서 임베딩 평균 계산
average_embedding = np.mean(document_embeddings, axis=0)

# 결과 확인
print(average_embedding)

# COMMAND ----------

from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd
review_text10 = review_text.limit(10)
# 스파크 데이터프레임을 판다스 데이터프레임으로 변환
pd_df = review_text10.toPandas()

# NLTK 다운로드
nltk.download('punkt')

# 문서를 토큰화하고 TaggedDocument 형식으로 변환
tagged_data = [TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)]) for i, review in enumerate(pd_df['reviewText'])]

# Doc2Vec 모델 학습
model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4, epochs=100)

# 문서 벡터 추출
document_embeddings = [model.dv[str(i)] for i in range(len(pd_df))]
print(document_embeddings)

# COMMAND ----------

import numpy as np

# 문서 임베딩 평균 계산
average_embedding = np.mean(document_embeddings, axis=0)

# 결과 확인
print(average_embedding)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - 전체 데이터 셋에 임베딩 값 저장하기
# MAGIC - vector_size = 100
# MAGIC - window = 2
# MAGIC - min_count = 10
# MAGIC - workers=4 
# MAGIC - epochs=100

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.config("spark.driver.memory", "4g").getOrCreate()

# COMMAND ----------

# MAGIC %pip install gensim

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)

# COMMAND ----------

from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd

review_text_pdf = review_text.toPandas()

# NLTK 다운로드
nltk.download('punkt')

# 문서를 토큰화하고 TaggedDocument 형식으로 변환
tagged_data = [TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)]) for i, review in enumerate(review_text_pdf['reviewText'])]

# Doc2Vec 모델 학습
model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=10, workers=4, epochs=100)

document_embeddings = [model.dv[str(i)] for i in range(len(pd_df))]

review_text_pdf['doc2vec'] = document_embeddings

# COMMAND ----------

from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd
from multiprocessing import Pool

# SparkSession 시작
spark = SparkSession.builder \
    .appName("Doc2Vec Example") \
    .getOrCreate()

# review_text를 Pandas DataFrame으로 변환
review_text_pdf = review_text.toPandas()

# NLTK 다운로드
nltk.download('punkt')

# 문서를 토큰화하고 TaggedDocument 형식으로 변환하는 함수
def preprocess_text(review):
    return TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)])

# 문서 전체를 멀티프로세스로 처리하여 TaggedDocument 형식으로 변환
with Pool(processes=4) as pool:
    tagged_data = pool.map(preprocess_text, review_text_pdf['reviewText'])

# Doc2Vec 모델 학습
model = Doc2Vec(vector_size=100, window=2, min_count=10, workers=4, epochs=100)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# 문서 벡터 추출
document_embeddings = [model.dv[str(i)] for i in range(len(review_text_pdf))]

# DataFrame에 문서 벡터 추가
review_text_pdf['doc2vec'] = document_embeddings

# SparkSession 종료
spark.stop()

# 변경된 DataFrame 확인
print(review_text_pdf.head())


# COMMAND ----------

from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd
import numpy as np

# SparkSession 시작
spark = SparkSession.builder \
    .appName("Doc2Vec Example") \
    .getOrCreate()

# review_text를 Pandas DataFrame으로 변환
review_text_pdf = review_text.toPandas()

# NLTK 다운로드
nltk.download('punkt')

# 제너레이터 함수 정의
def generate_batches(data, batch_size=100):
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        yield batch

# 문서 벡터 추출 함수 정의
def extract_document_embeddings(batch):
    tagged_data = [TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)]) for i, review in enumerate(batch['reviewText'])]
    model = Doc2Vec(vector_size=100, window=2, min_count=10, workers=4, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    document_embeddings = [model.dv[str(i)] for i in range(len(batch))]
    return document_embeddings

# 제너레이터를 사용하여 문서 벡터 추출
batch_size = 100
document_embeddings = []
for batch in generate_batches(review_text_pdf, batch_size):
    batch_embeddings = extract_document_embeddings(batch)
    document_embeddings.extend(batch_embeddings)

# DataFrame에 문서 벡터 추가
review_text_pdf['doc2vec'] = document_embeddings

# SparkSession 종료
spark.stop()

# COMMAND ----------

print(review_text_pdf.head())

# COMMAND ----------

!pip install ray

# COMMAND ----------

# MAGIC %pip install gensim

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

review_text = spark.read.table("asac.senti_review_text")

# COMMAND ----------

review_text_nona = review_text.dropna(subset=['reviewText'])

# COMMAND ----------

import ray
from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd

# SparkSession 시작
spark = SparkSession.builder \
    .appName("Doc2Vec Example") \
    .getOrCreate()

review_text_pdf = review_text_nona.toPandas()

# NLTK 다운로드
nltk.download('punkt')

# Ray 초기화
ray.init()

# 문서 벡터 추출을 위한 Actor 클래스 정의
@ray.remote
class Doc2VecActor:
    def __init__(self):
        self.model = None
    
    def preprocess_text(self, review):
        return TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)])
    
    def train_model(self, tagged_data):
        self.model = Doc2Vec(vector_size=100, window=2, min_count=10, workers=4, epochs=100)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    
    def extract_document_embeddings(self, batch):
        tagged_data = ray.get(batch)
        document_embeddings = [self.model.dv[str(i)] for i in range(len(tagged_data))]
        return document_embeddings

# Doc2VecActor 객체 생성
doc2vec_actor = Doc2VecActor.remote()

# 문서를 토큰화하고 TaggedDocument 형식으로 변환
tagged_data = [TaggedDocument(words=word_tokenize(review.lower()), tags=[str(i)]) for i, review in enumerate(review_text_pdf['reviewText'])]

# tagged_data를 Ray Object로 변환
tagged_data_ray = ray.put(tagged_data)

# 모델 학습
ray.get(doc2vec_actor.train_model.remote(tagged_data_ray))

# 문서 벡터 추출
document_embeddings_ray = ray.get(doc2vec_actor.extract_document_embeddings.remote(tagged_data_ray))

# DataFrame에 문서 벡터 추가
review_text_pdf['doc2vec'] = document_embeddings_ray

# SparkSession 종료
spark.stop()

# Ray 종료
ray.shutdown()

# 변경된 DataFrame 확인
print(review_text_pdf.head())


# COMMAND ----------

!pip install ray
%pip install gensim


# COMMAND ----------

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt
import ray
from pyspark.sql import SparkSession
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import pandas as pd
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import monotonically_increasing_id

nltk.download('punkt')

review_text_nona = review_text.dropna(subset=['reviewText'])

# Tokenizer를 사용하여 문장을 토큰화
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
words_df = tokenizer.transform(review_text_nona)

word2vec = Word2Vec(vectorSize=100, minCount=10, inputCol="words", outputCol="doc2vec")
model = word2vec.fit(words_df)

# 문서 벡터 추출
document_embeddings = model.transform(words_df)
document_embeddings.show()


# COMMAND ----------

num_partitions = 4
name = "asac.document_embeddings"

document_embeddings.repartition(num_partitions).write.saveAsTable(name, mode="overwrite")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Spark NLP 모델 사용하기
# MAGIC - Smaller BERT Sentence Embeddings (L-10_H-128_A-2)
# MAGIC - https://sparknlp.org/2020/08/25/sent_small_bert_L10_128.html

# COMMAND ----------

# MAGIC %md
# MAGIC - https://tacademykr-daanalysis.cloud.databricks.com/?o=647747681770278#notebook/738975682411579/command/738975682411580

# COMMAND ----------

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)
review_text_nona = review_text.dropna(subset=['reviewText'])

# COMMAND ----------

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings, AlbertEmbeddings, Tokenizer, Normalizer, StopWordsCleaner, RoBertaSentenceEmbeddings
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, DoubleType
import numpy as np
from pyspark.sql.functions import col, size, array, expr
dot_udf = F.udf(lambda x,y: float(x.dot(y)) / float(x.norm(2)*y.norm(2)), DoubleType())

import operator
import datetime as dt
from pyspark.ml.feature import Word2Vec, Word2VecModel, Normalizer
from pyspark.sql.functions import format_number as fmt
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from pyspark.ml.linalg import Vectors

from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType
dot_udf = F.udf(lambda x,y: float(x.dot(y)) / float(x.norm(2)*y.norm(2)), DoubleType())

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cosine_similarity_udf = udf(cosine_similarity, FloatType())

import pyspark.sql.functions as F
# define udf
def sorter(l):
  res = sorted(l, key=operator.itemgetter(0))
  return [item[1] for item in res]

sort_udf = F.udf(sorter)

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")
  
embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128", "en") \
    .setInputCols(["sentences"]) \
    .setOutputCol("sentence_bert_embeddings")\
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512)

# embeddings = AlbertEmbeddings.pretrained("albert_base_uncased", "en") \
# .setInputCols("sentences", "token") \
# .setOutputCol("embeddings")

# COMMAND ----------

nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, embeddings])

# COMMAND ----------

model = nlp_pipeline.fit(review_text_nona)
result_embeddings = model.transform(review_text_nona)

# COMMAND ----------

display(result_embeddings)

# COMMAND ----------

display(result_embeddings)
# 문장당 128차원의 벡터 생성

# COMMAND ----------

num_partitions = 4
name = "asac.result_embeddings_sbert"
result_embeddings.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

# result_embeddings.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/nlp/sent_small_bert_L2_128')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.result_embeddings_sbert

# COMMAND ----------

# MAGIC %md
# MAGIC #### 임베딩 다루기
# MAGIC - 우선 한 문장당 128차원의 임베딩 값들이 생성 -> 문장통합 -> asin 통합 필요
# MAGIC - sentence_bert_embeddings열에서 순수하게 임베딩 값만 추출하는 것 필요
# MAGIC -> 이걸 embedding 열로 저장했다고 가정
# MAGIC - 평균 / pca / t-SNE / MDS
# MAGIC - Autoencoders / LLE 따로 구현 필요
# MAGIC - clustering / Factor Analysis

# COMMAND ----------

# 평균
from pyspark.sql.functions import col, expr

# 'embedding' 열의 값들을 평균내어 새로운 열 추가
result_with_mean = result_embeddings.withColumn("mean_embedding", expr("aggregate(embedding, cast(0.0 as double), (acc, x) -> acc + x) / size(embedding)"))


# COMMAND ----------

data = [("ASIN1", [1, 2, 3]), ("ASIN1", [4, 5, 6]), ("ASIN2", [7, 8, 9]), ("ASIN2", [10, 11, 12])]
df = spark.createDataFrame(data, ["asin", "scores"])
array_length = 128

# 배열 평균
def average_arrays(arrays):
    sum_arrays = [0] * array_length
    for arr in arrays:
        sum_arrays = [sum(x) for x in zip(sum_arrays, arr + [0]*(array_length - len(arr)))]
    return [x / len(arrays) for x in sum_arrays]

spark.udf.register("average_arrays_udf", average_arrays, ArrayType(DoubleType()))

# 각 ASIN별로 배열의 평균 계산
result = df.groupBy("asin").agg(expr("average_arrays_udf(collect_list(scores))").alias("average_scores"))

# COMMAND ----------

# pca 방법
from pyspark.ml.feature import PCA

pca = PCA(k=10, inputCol="embedding", outputCol="pca_features")
pca_model = pca.fit(result_embeddings)
result = pca_model.transform(result_embeddings)

# COMMAND ----------

# t-SNE
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import TSNE

assembler = VectorAssembler(inputCols=["embedding"], outputCol="features")
data_vectorized = assembler.transform(result_embeddings)

# 데이터 스케일링
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
data_scaled = scaler.fit(data_vectorized).transform(data_vectorized)

# t-SNE 모델 생성 및 변환
tsne = TSNE(k=10, inputCol="scaled_features", outputCol="tsne_features")
tsne_model = tsne.fit(data_scaled)
result_tsne = tsne_model.transform(data_scaled)

# COMMAND ----------

# MDS
from pyspark.ml.feature import MDS

mds = MDS(k=10, inputCol="embedding", outputCol="mds_features")
mds_model = mds.fit(result_embeddings)
result_mds = mds_model.transform(result_embeddings)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. 허깅페이스 모델 사용하기 (이건 안해도 됨)
# MAGIC - https://huggingface.co/sentence-transformers

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


