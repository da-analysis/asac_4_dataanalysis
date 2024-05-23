# Databricks notebook source
# MAGIC %pip install plotly

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

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)

# COMMAND ----------

review_text_10 =review_text.limit(10)
review_text_nona = review_text.dropna(subset=['reviewText'])
review_text_pdf = review_text_nona.toPandas()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Doc2vec 모델 (델타테이블 저장 완료)
# MAGIC https://sparknlp.org/2021/11/21/doc2vec_gigaword_wiki_300_en.html

# COMMAND ----------

from pyspark.ml import Pipeline

document = DocumentAssembler()\
.setInputCol("reviewText")\
.setOutputCol("document")

token = Tokenizer()\
.setInputCols("document")\
.setOutputCol("token")

norm = Normalizer()\
.setInputCols(["token"])\
.setOutputCol("normalized")\
.setLowercase(True)

stops = StopWordsCleaner.pretrained()\
.setInputCols("normalized")\
.setOutputCol("cleanedToken")

doc2Vec = Doc2VecModel.pretrained("doc2vec_gigaword_wiki_300", "en")\
.setInputCols("cleanedToken")\
.setOutputCol("sentence_embeddings")


# COMMAND ----------

# MAGIC %md
# MAGIC 10개만 돌려보기

# COMMAND ----------

doc_pipeline = Pipeline(stages=[document, token,norm,stops,doc2Vec])
model = doc_pipeline.fit(review_text_10)
result_embeddings = model.transform(review_text_10)

# COMMAND ----------

display(result_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC 판다스로 변경해서 돌려보기

# COMMAND ----------

doc_pipeline = Pipeline(stages=[document, token,norm,stops,doc2Vec])
model_pd = doc_pipeline.fit(review_text_pdf)
result_embeddings = model_pd.transform(review_text_pdf)

# COMMAND ----------

doc_pipeline = Pipeline(stages=[document, token,norm,stops,doc2Vec])
model = doc_pipeline.fit(review_text_nona)
result_embeddings = model.transform(review_text_nona)

# COMMAND ----------

display(result_embeddings.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC 원본에서 na행 삭제한 뒤 전체 저장하기

# COMMAND ----------

name = "asac.result_embeddings_doc2vec"
result_embeddings.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.result_embeddings_doc2vec

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.result_embeddings_doc2vec

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 임베딩 결과 대표화 하기

# COMMAND ----------

doc2vec = spark.read.table("asac.result_embeddings_doc2vec")
doc2vec1000 = doc2vec.limit(1000)

name = "asac.result_embeddings_doc2vec1000"
doc2vec1000.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

doc2vec1000 = spark.read.table("asac.result_embeddings_doc2vec1000")

# COMMAND ----------

display(doc2vec1000)

# COMMAND ----------

from pyspark.sql.functions import col, expr

# embeddings 값 추출
def extract_embeddings(embeddings):
    return [entry['embeddings'] for entry in embeddings]

# emb컬럼으로 생성함
doc2vec1000 = doc2vec1000.withColumn("emb", expr("transform(sentence_embeddings, x -> x['embeddings'])"))

# COMMAND ----------

display(doc2vec1000.limit(100))

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, count(*) cnt  from asac.result_embeddings_doc2vec1000
# MAGIC group by asin
# MAGIC order by cnt desc

# COMMAND ----------

from pyspark.sql.functions import expr

# new_emb 열의 차원을 줄이기 위해 flatMap 함수를 사용하여 각 행의 리스트를 하나의 리스트로 변환
doc2vec1000 = doc2vec1000.withColumn("new_emb", expr("flatten(emb)"))


# COMMAND ----------

display(doc2vec1000)

# COMMAND ----------

from pyspark.sql.functions import collect_list

new_doc2vec1000 = doc2vec1000.groupBy("asin") \
                             .agg(collect_list("emb").alias("new_emb_2"))

# COMMAND ----------

display(new_doc2vec1000)

# COMMAND ----------

from pyspark.sql.functions import expr

# new_emb 열의 차원을 줄이기 위해 flatMap 함수를 사용하여 각 행의 리스트를 하나의 리스트로 변환
new_doc2vec1000 = new_doc2vec1000.withColumn("new_emb_flat", expr("flatten(new_emb_2)"))

# COMMAND ----------

display(new_doc2vec1000)

# COMMAND ----------

from pyspark.sql.functions import posexplode, col, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA

# asin 열을 포함하여 explode하고, array를 vector로 변환하는 과정을 진행
df_exploded_with_asin = new_doc2vec1000.select("asin", posexplode("new_emb_flat").alias("pos", "array"))

# UDF를 사용하여 array를 dense vector로 변환
to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
df_vector_with_asin = df_exploded_with_asin.withColumn("features", to_vector(col("array")))

# PCA 모델 생성 및 적용
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df_vector_with_asin)
result_pca_with_asin = model.transform(df_vector_with_asin).select("asin", "pos", "pcaFeatures")

# 결과 확인
result_pca_with_asin.show()


# COMMAND ----------

display(result_pca_with_asin)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# DenseVector에서 값을 추출
extract_vector_values = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))

# pcaFeatures 열에서 값들만 추출하여 새로운 열에 저장
result_pca_with_values = result_pca_with_asin.withColumn("pcaValues", extract_vector_values("pcaFeatures"))

display(result_pca_with_values)

# COMMAND ----------

import plotly.graph_objects as go
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import hashlib


pandas_df = result_pca_with_values.toPandas()

# 특정 asin 값에 대해서만 데이터 필터링
filtered_df = pandas_df[pandas_df['asin'].isin(['B000IEAYN6', 'B003EW81Q6','962886436X'])]

# asin 값에 따라 고유한 색상 코드 생성
def generate_color_code(asin):
    hash_object = hashlib.md5(asin.encode())
    return '#' + hash_object.hexdigest()[:6]

filtered_df['color'] = filtered_df['asin'].apply(generate_color_code)

# 3D 플롯 생성
fig = go.Figure()



for index, row in filtered_df.iterrows():
    fig.add_trace(go.Scatter3d(x=[row['pcaValues'][0]],
                               y=[row['pcaValues'][1]],
                               z=[row['pcaValues'][2]],
                               mode='markers',
                               marker=dict(size=5, color=row['color']),
                               text=row['asin'],  
                               name=row['asin']))

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

# COMMAND ----------

from pyspark.sql import functions as F
max_value = result_pca_with_values.agg(F.max("pos").alias("max_pos")).collect()[0]["max_pos"]
from pyspark.sql import functions as F
avg_value = result_pca_with_values.agg(F.avg("pos").alias("avg_pos")).collect()[0]["avg_pos"]
median_value = result_pca_with_values.agg(F.median("pos").alias("median_pos")).collect()[0]["median_pos"]

print("max:",max_value,"avg:",avg_value,"median:",median_value)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark S-bert sentenece (델타테이블 저장 완료)
# MAGIC - Smaller BERT Sentence Embeddings (L-10_H-128_A-2)
# MAGIC - https://sparknlp.org/2020/08/25/sent_small_bert_L10_128.html
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
model = nlp_pipeline.fit(review_text_nona)
result_embeddings = model.transform(review_text_nona)

# COMMAND ----------

display(result_embeddings) # 문장당 128차원의 벡터 생성

# COMMAND ----------

name = "asac.result_embeddings_sbert"
result_embeddings.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.result_embeddings_sbert

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.result_embeddings_sbert

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 임베딩 결과 대표화 하기

# COMMAND ----------

sbert = spark.read.table("asac.result_embeddings_sbert")

# COMMAND ----------

# 1000개만 별로 테이블로 저장해보기
sbert1000 = sbert.limit(1000)

# COMMAND ----------

name = "asac.result_embeddings_sbert1000"
sbert1000.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------

sbert1000 = spark.read.table("asac.result_embeddings_sbert1000")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.result_embeddings_sbert1000

# COMMAND ----------

from pyspark.sql.functions import col, expr

# embeddings 값 추출
def extract_embeddings(embeddings):
    return [entry['embeddings'] for entry in embeddings]

# emb컬럼으로 생성함
sbert1000 = sbert1000.withColumn("emb", expr("transform(sentence_bert_embeddings, x -> x['embeddings'])"))

# COMMAND ----------

display(sbert1000.limit(100))

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct(asin)) from asac.result_embeddings_sbert1000

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, count(*) cnt  from asac.result_embeddings_sbert1000
# MAGIC group by asin
# MAGIC order by cnt desc

# COMMAND ----------

from pyspark.sql.functions import collect_list

new_sbert = sbert1000.groupBy("asin") \
                             .agg(collect_list("emb").alias("new_emb"))

# COMMAND ----------

display(new_sbert.limit(1))

# COMMAND ----------

display(new_sbert)

# COMMAND ----------

display(sbert1000.filter(sbert1000.asin == "B00MXWFUQC"))

# COMMAND ----------

display(new_sbert.filter(sbert1000.asin == "B00MXWFUQC"))

# COMMAND ----------

from pyspark.sql.functions import expr

# new_emb 열의 차원을 줄이기 위해 flatMap 함수를 사용하여 각 행의 리스트를 하나의 리스트로 변환
new_sbert = new_sbert.withColumn("new_emb_flat", expr("flatten(new_emb)"))


# COMMAND ----------

display(new_sbert.filter(sbert1000.asin == "B00MXWFUQC"))

# COMMAND ----------

from pyspark.sql.functions import posexplode, col, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA

# asin 열을 포함하여 explode하고, array를 vector로 변환하는 과정을 진행
df_exploded_with_asin = new_sbert.select("asin", posexplode("new_emb_flat").alias("pos", "array"))

# UDF를 사용하여 array를 dense vector로 변환
to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
df_vector_with_asin = df_exploded_with_asin.withColumn("features", to_vector(col("array")))

# PCA 모델 생성 및 적용
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df_vector_with_asin)
result_pca_with_asin = model.transform(df_vector_with_asin).select("asin", "pos", "pcaFeatures")

# 결과 확인
result_pca_with_asin.show()


# COMMAND ----------

display(result_pca_with_asin)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# DenseVector에서 값을 추출
extract_vector_values = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))

# pcaFeatures 열에서 값들만 추출하여 새로운 열에 저장
result_pca_with_values = result_pca_with_asin.withColumn("pcaValues", extract_vector_values("pcaFeatures"))

display(result_pca_with_values)

# COMMAND ----------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import hashlib


pandas_df = result_pca_with_values.toPandas()

# asin 값을 기반으로 고유한 색상 코드 생성
def generate_color_code(asin):
    hash_object = hashlib.md5(asin.encode())
    return '#' + hash_object.hexdigest()[:6]

pandas_df['color'] = pandas_df['asin'].apply(generate_color_code)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 각 점을 3D 공간에 플롯
for index, row in pandas_df.iterrows():
    ax.scatter(row['pcaValues'][0], row['pcaValues'][1], row['pcaValues'][2], color=row['color'])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

plt.show()

# COMMAND ----------

# 특정 asin 값에 대해서만 데이터 필터링
filtered_df = pandas_df[pandas_df['asin'].isin(['B000IEAYN6', 'B003EW81Q6','962886436X'])]

# asin 값에 따라 고유한 색상 코드 생성
def generate_color_code(asin):
    hash_object = hashlib.md5(asin.encode())
    return '#' + hash_object.hexdigest()[:6]

filtered_df['color'] = filtered_df['asin'].apply(generate_color_code)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


for index, row in filtered_df.iterrows():
    ax.scatter(row['pcaValues'][0], row['pcaValues'][1], row['pcaValues'][2], color=row['color'], label=row['asin'])

# 범례 추가 (중복된 항목 제거를 위해)
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

plt.show()

# COMMAND ----------

import plotly.graph_objects as go
import pandas as pd

# 3D 스캐터 플롯 생성
fig = go.Figure()

for index, row in filtered_df.iterrows():
    fig.add_trace(go.Scatter3d(x=[row['pcaValues'][0]],
                               y=[row['pcaValues'][1]],
                               z=[row['pcaValues'][2]],
                               mode='markers',
                               marker=dict(size=5, color=row['color']),
                               text=row['asin'],  
                               name=row['asin']))

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

# COMMAND ----------

from pyspark.sql import functions as F
max_value = result_pca_with_values.agg(F.max("pos").alias("max_pos")).collect()[0]["max_pos"]
from pyspark.sql import functions as F
avg_value = result_pca_with_values.agg(F.avg("pos").alias("avg_pos")).collect()[0]["avg_pos"]
median_value = result_pca_with_values.agg(F.median("pos").alias("median_pos")).collect()[0]["median_pos"]

print("max:",max_value,"avg:",avg_value,"median:",median_value)

# COMMAND ----------

## asin끼리 묶었음
from pyspark.sql.functions import collect_list

new_result_pca_with_values = result_pca_with_values.groupBy("asin") \
                             .agg(collect_list("pcaValues").alias("new_pcaValues"))

# COMMAND ----------

display(new_result_pca_with_values)

# COMMAND ----------

# Assuming this DataFrame is already defined
pandas_df = new_result_pca_with_values.toPandas()

# Function to calculate centroids for lists of PCA values
def calculate_centroids(pcaValues):
    centroids = []
    for sublist in pcaValues:
        try:
            # First, ensure sublist is an array to simplify shape checks
            sublist_arr = np.array(sublist)
            # Check if sublist_arr is indeed 2-dimensional
            if sublist_arr.ndim == 2 and sublist_arr.shape[1] > 0:
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=2, random_state=0).fit(sublist_arr)
                # Append the computed centroids
                centroids.append(kmeans.cluster_centers_.tolist())
        except Exception as e:
            # Optionally handle or log the error for this sublist
            # For now, just pass to skip invalid entries
            pass
    return centroids

new_columns = pandas_df['new_pcaValues'].apply(calculate_centroids)

# Concatenate these new columns to your original DataFrame
pandas_df = pd.concat([pandas_df, new_columns], axis=1)

# COMMAND ----------

pandas_df

# COMMAND ----------

new_result_pca_with_values_1 = new_result_pca_with_values.limit(1)

# COMMAND ----------

pandas_df = new_result_pca_with_values.toPandas()

# COMMAND ----------

pd.set_option('display.max_colwidth', None)
pandas_df.head(5)

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import numpy as np


# VectorAssembler를 사용하여 각 벡터를 단일 벡터로 변환
vector_assembler = VectorAssembler(inputCols=["new_pcaValues"], outputCol="features")
df_assembled = vector_assembler.transform(new_result_pca_with_values_1)

# KMeans 클러스터링 모델 초기화
kmeans = KMeans(k=2, seed=1)

# 모델을 데이터에 적합시키고 예측 수행
model = kmeans.fit(df_assembled)
predictions = model.transform(df_assembled)

# 예측 결과를 출력
predictions.select("asin", "prediction").show()


# COMMAND ----------

# Explode the new_pcaValues array to treat each element as a row
df_exploded = new_result_pca_with_values.withColumn("new_pcaValues", explode(new_result_pca_with_values.new_pcaValues))

# Apply the perform_conditional_kmeans function
result = df_exploded.groupby('asin').apply(perform_conditional_kmeans)

result.show()


# COMMAND ----------

# MAGIC %md
# MAGIC - 클러스터링으로 각각 2개의 리스트를 만들고 싶었는데, 아무리 해도 계속 에러 발생
# MAGIC - 우선 3차원으로 줄인 리스트들의 평균을 임베딩 값으로 활용

# COMMAND ----------

# 평균
display(new_result_pca_with_values)

# COMMAND ----------

# 여러 배열의 평균을 계산하는 함수
def average_of_arrays(*arrays):
    if len(arrays) == 0:
        return []
    
    array_length = len(arrays[0])
    for arr in arrays:
        if len(arr) != array_length:
            raise ValueError("All arrays must have the same length")
    
    return [sum(arr[i] for arr in arrays) / len(arrays) for i in range(array_length)]
average_of_arrays_udf = udf(average_of_arrays, ArrayType(DoubleType()))

df = spark.createDataFrame([([1, 2, 3], [4, 5, 6],), ([7, 8, 9], [10, 11, 12],)], ['array1', 'array2'])

df.withColumn("average", average_of_arrays_udf(df["array1"], df["array2"])).show()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession

# SparkSession 초기화
spark = SparkSession.builder.appName("example").getOrCreate()

# 예시 데이터 생성
data = [("B000IEAYN6", [[0.1, 0.2, 0.1], [0.1, 0.2, 0.3]]),
        ("B003EW81Q6", [[0.1, 0.2, 0.1], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])]
columns = ["asin", "new_pcaValues"]

df = spark.createDataFrame(data, schema=columns)

# 평균 벡터 계산 함수
def calculate_mean(vectors):
    arr = np.array(vectors)
    mean_vector = arr.mean(axis=0)
    mean_vector_rounded = np.round_(mean_vector, 4)
    return mean_vector_rounded.tolist()

calculate_mean_udf = udf(calculate_mean, ArrayType(DoubleType()))

new_df = df.withColumn("new_emd_mean", calculate_mean_udf("new_pcaValues"))

display(new_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

# 평균 벡터 계산 함수
def calculate_mean(vectors):
    arr = np.array(vectors)
    mean_vector = arr.mean(axis=0)
    mean_vector_rounded = np.round_(mean_vector, 4)
    return mean_vector_rounded.tolist()

calculate_mean_udf = udf(calculate_mean, ArrayType(DoubleType()))

new_df = new_result_pca_with_values.withColumn("new_emd_mean", calculate_mean_udf("new_pcaValues"))

display(new_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 코사인 유사도

# COMMAND ----------

from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType
dot_udf = F.udf(lambda x,y: float(x.dot(y)) / float(x.norm(2)*y.norm(2)), DoubleType())

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cosine_similarity_udf = udf(cosine_similarity, FloatType())

#####################
bert_cf = bert_cf.withColumn('dot',   F.expr('aggregate(arrays_zip(asin1_vector, asin2_vector), 0D, (acc, x) -> acc + (x.asin1_vector * x.asin2_vector))')) \
  .withColumn('norm1', F.expr('sqrt(aggregate(asin1_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('norm2', F.expr('sqrt(aggregate(asin2_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('cos_vector', F.expr('dot / (norm1 * norm2)'))


# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import col
from pyspark.sql.functions import expr
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

target_row = new_df.filter(col("asin") == "B00063E4KS").select("new_emd_mean").collect()[0][0]

# 코사인 유사도 계산
def cosine_similarity(v1, v2):
    dot_product = float(v1.dot(v2))
    norm_v1 = float(v1.norm(2))
    norm_v2 = float(v2.norm(2))
    return dot_product / (norm_v1 * norm_v2)

cosine_similarity_udf = udf(lambda x: cosine_similarity(Vectors.dense(target_row), Vectors.dense(x)), DoubleType())
result_df = new_df.withColumn("cosine_similarity", cosine_similarity_udf(col("new_emd_mean")))
result_df = result_df.filter(col("asin") != "B00063E4KS")

# 코사인 유사도를 기준으로 내림차순 정렬
windowSpec = Window.orderBy(col("cosine_similarity").desc())
result_df_cos = result_df.withColumn("rank", row_number().over(windowSpec))

display(result_df_cos)


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B00063E4KS" 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B007JY6WAW"

# COMMAND ----------

# MAGIC %md
# MAGIC 유클리드 거리

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import col
from pyspark.sql.functions import expr
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

target_row = new_df.filter(col("asin") == "B00063E4KS").select("new_emd_mean").collect()[0][0]

# 유클리드 거리 계산
def euclidean_distance(v1, v2):
    return float(v1.squared_distance(v2))

euclidean_distance_udf = udf(lambda x: euclidean_distance(Vectors.dense(target_row), Vectors.dense(x)), DoubleType())
result_df = new_df.withColumn("euclidean_distance", euclidean_distance_udf(col("new_emd_mean")))
result_df = result_df.filter(col("asin") != "B00063E4KS")

# 유클리드 거리를 기준으로 오름차순 정렬
windowSpec = Window.orderBy(col("euclidean_distance").asc())
result_df_euc = result_df.withColumn("rank", row_number().over(windowSpec))

display(result_df_euc)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B00063E4KS" 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B00M0UBAKC" 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 맨하튼거리

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

target_row = new_df.filter(col("asin") == "B00063E4KS").select("new_emd_mean").collect()[0][0]

# 맨하탄 거리 계산 함수 정의
def manhattan_distance(v1, v2):
    return float(sum(abs(x - y) for x, y in zip(v1, v2)))

manhattan_distance_udf = udf(lambda x: manhattan_distance(target_row, x), DoubleType())
result_df = new_df.withColumn("manhattan_distance", manhattan_distance_udf(col("new_emd_mean")))
result_df = result_df.filter(col("asin") != "B00063E4KS")

# 맨하탄 거리를 기준으로 오름차순 정렬
windowSpec = Window.orderBy(col("manhattan_distance").asc())
result_df_man = result_df.withColumn("rank", row_number().over(windowSpec))

display(result_df_man)


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B00063E4KS" 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.senti_review_text
# MAGIC where asin == "B00M0UBAKC" 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 허깅페이스 sentence-transformers all-MiniLM-L6-v2 
# MAGIC - 앞에 임베딩 모델 2개 있으니까 우선 안해도 될듯
# MAGIC - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# MAGIC - 90MB (코드 있지만, pytorch 활용)
# MAGIC
# MAGIC - https://huggingface.co/Shimiao/all-MiniLM-L6-v2-finetuned-wikitext2 (코드 없음)
# MAGIC - https://sparknlp.org/2023/09/13/all_minilm_l6_v2_finetuned_wikitext2_en.html (코드 생략됨)
# MAGIC - 84.6MB

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

# COMMAND ----------



# COMMAND ----------

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings, AlbertEmbeddings, Tokenizer, Normalizer, StopWordsCleaner, RoBertaSentenceEmbeddings
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import BertEmbeddings
from pyspark.ml import Pipeline


# COMMAND ----------

review_text = spark.read.table("asac.senti_review_text")
# review_text = ps.DataFrame(review_text)
review_text_nona = review_text.dropna(subset=['reviewText'])

# COMMAND ----------

document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("documents")
    
tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")
    
embeddings =BertEmbeddings.pretrained("all_minilm_l6_v2_finetuned_wikitext2","en") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("embeddings")


# COMMAND ----------

pipeline = Pipeline().setStages([document_assembler, embeddings])
pipelineModel = pipeline.fit(review_text_nona)
pipelineDF = pipelineModel.transform(review_text_nona)

# COMMAND ----------

display(pipelineDF.limit(10)) 

# COMMAND ----------

name = "asac.result_embeddings_mini"
pipelineDF.write.saveAsTable(name, mode="overwrite")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# https://stackoverflow.com/questions/60492839/how-to-compare-sentence-similarities-using-embeddings-from-bert

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


