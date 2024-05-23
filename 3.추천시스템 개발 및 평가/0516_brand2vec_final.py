# Databricks notebook source
# MAGIC %pip install pandas

# COMMAND ----------

import operator
import datetime as dt
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel, Normalizer
from pyspark.sql.functions import format_number as fmt
from pyspark.sql.types import FloatType, DoubleType
import numpy as np
from pyspark.ml.linalg import Vectors
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

amazon_data = spark.read.table("asac.brand2vec")

# COMMAND ----------

grouped_df = (
amazon_data
.groupby("reviewerID")
.agg(
F.sort_array(F.collect_list(F.struct("reviewTime", "brand"))) 
.alias("collected_list")
)
.withColumn("sorted_list", F.col("collected_list.brand"))
.drop("collected_list")
)

# COMMAND ----------

display(grouped_df)

# COMMAND ----------

grouped_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### reviewerID별로 브랜드 개수 분포 확인하기
# MAGIC - list_len 칼럼 생성
# MAGIC - 그래프로 그리기 (density, frequency)

# COMMAND ----------

from pyspark.sql.functions import size

grouped_df = grouped_df.withColumn("list_len", size("sorted_list"))

# COMMAND ----------

display(grouped_df)

# COMMAND ----------

# 6210748
len5 = grouped_df.filter(col("list_len")<5)
print(len5.count())
print(len5.count()/grouped_df.count())

# COMMAND ----------

grouped_df.count() - len5.count()

# COMMAND ----------

from pyspark.sql.functions import count
list_len_counts = grouped_df.groupBy("list_len").count()
list_len_counts.show()

# COMMAND ----------

display(list_len_counts)

# COMMAND ----------

grouped_df_5 = grouped_df.filter(col("list_len")>=5)
display(grouped_df_5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 브랜드투벡 모델
# MAGIC - 리뷰어별로 5개이상 브랜드 갖는 데이터만 활용(grouped_df_5)
# MAGIC - 전체는 grouped_df

# COMMAND ----------

word2Vec = Word2Vec(vectorSize=30, seed=1004, inputCol="sorted_list", outputCol="model", minCount=10, maxIter = 30, numPartitions = 16, windowSize = 5) 
model = word2Vec.fit(grouped_df_5)

# 원래 모델 path
# vectorSize=30, seed=1004, inputCol="sorted_list", outputCol="model", minCount=10, maxIter = 30, numPartitions = 16, windowSize = 5
#model_path = "dbfs:/FileStore/amazon/model/brand2vec"
#model.write().overwrite().save(model_path)

model_path2 = "dbfs:/FileStore/amazon/model/brand2vec_ver2"
model.write().overwrite().save(model_path2)

# COMMAND ----------

model = Word2VecModel.load("dbfs:/FileStore/amazon/model/brand2vec_ver2")

# COMMAND ----------

from pyspark.sql.functions import format_number as fmt
display(model.findSynonyms("Nokia", 10).select("word", fmt("similarity", 5).alias("similarity")))

# COMMAND ----------

from pyspark.sql.functions import format_number as fmt
display(model.findSynonyms("APPLE", 10).select("word", fmt("similarity", 5).alias("similarity")))

# COMMAND ----------

brand_vec = model.getVectors()
brand_vec.createOrReplaceTempView("brand_vec")
display(brand_vec)

# COMMAND ----------

name = "asac.brand2vec_emb_2"
brand_vec.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 조인 전 데이터

# COMMAND ----------

brand_vec = spark.read.table("asac.brand2vec_emb_2")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb_2

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand2vec_emb_2

# COMMAND ----------

df_cross_joined = brand_vec.crossJoin(brand_vec)

# COMMAND ----------

from pyspark.sql.functions import col

df1 = brand_vec.withColumnRenamed("word", "brand_1")
df1 = df1.withColumnRenamed("vector", "vector_1")
df2 = brand_vec.withColumnRenamed("word", "brand_2")
df2 = df2.withColumnRenamed("vector", "vector_2")

# 동일한 데이터프레임을 복제하여 교차 조인 실행
df_cross_joined = df1.crossJoin(df2)

# 동일한 브랜드 조합을 필터링하여 제거
df_filtered = df_cross_joined.filter(col("brand_1") != col("brand_2"))

# 결과 출력
df_filtered.show(1)

# COMMAND ----------

name = "asac.brand2vec_emb_2_cos_1"
df_filtered.write.saveAsTable(name)

# COMMAND ----------

df_filtered = spark.read.table("asac.brand2vec_emb_2_cos_1")

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.linalg import DenseVector

# PySpark UDF 정의
def dense_vector_to_list(dv):
    # DenseVector를 파이썬 리스트로 변환
    return dv.toArray().tolist()

# UDF 등록
to_list_udf = F.udf(dense_vector_to_list, T.ArrayType(T.DoubleType()))

# 'vector_1'과 'vector_2' 열을 리스트로 변환
df_filtered = df_filtered.withColumn("emb1", to_list_udf(F.col("vector_1")))
df_filtered = df_filtered.withColumn("emb2", to_list_udf(F.col("vector_2")))

# 원래의 embedding 열 삭제
df_filtered = df_filtered.drop("vector_1")
df_filtered = df_filtered.drop("vector_2")

# 스키마 데이터 출력 및 데이터 프레임 표시
df_filtered.printSchema()
display(df_filtered)


# COMMAND ----------

name = "asac.brand2vec_emb_2_cos_2"
df_filtered.write.saveAsTable(name)

# COMMAND ----------

df_filtered = spark.read.table("asac.brand2vec_emb_2_cos_2")

# COMMAND ----------

# 방법2
import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf, PandasUDFType

columns = [("emb1", "emb2")]

@pandas_udf("double", PandasUDFType.SCALAR)
def cosine_similarity_udf(v1: pd.Series, v2: pd.Series) -> pd.Series:
    # 각 Series의 요소가 벡터인 경우를 처리하기 위한 수정
    dot_product = np.array([np.dot(a, b) for a, b in zip(v1, v2)])
    norm_v1 = np.sqrt(np.array([np.dot(a, a) for a in v1]))
    norm_v2 = np.sqrt(np.array([np.dot(b, b) for b in v2]))
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return pd.Series(cosine_similarity)


# DataFrame에 코사인 유사도 컬럼 추가
from pyspark.sql.functions import col, when

for col1, col2 in columns:
    df_cos = df_filtered.withColumn(
        f"{col1[:-2]}_cosine_similarity",
        when(
            col(col1).isNull() | col(col2).isNull(),
            0
        ).otherwise(cosine_similarity_udf(col(col1), col(col2)))
    )

# COMMAND ----------

name = "asac.brand2vec_emb_2_cos_3"
df_cos.write.saveAsTable(name)

# COMMAND ----------

df_cos = spark.read.table("asac.brand2vec_emb_2_cos_3")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 중카 붙이기

# COMMAND ----------

cat2_data = spark.sql("""
select brand, cat2
from asac.meta_cell_phones_and_accessories_new_price2
""")

# COMMAND ----------

brand_vectors = spark.read.table("asac.brand2vec_emb_2")

# COMMAND ----------

cat2_data = cat2_data.withColumnRenamed("brand", "word")
brand_vectors_cat = brand_vectors.join(cat2_data, on="word", how="left")

# COMMAND ----------

name = "asac.brand_vectors_2_cat"
brand_vectors_cat.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand_vectors_2_cat

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### t-sne 구현하기
# MAGIC - https://www.databricks.com/notebooks/reprisk_notebooks/

# COMMAND ----------

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
brand_vectors = spark.read.table("asac.brand2vec_emb_2")
brand_vectors_pandas = brand_vectors.toPandas()

import numpy as np


# Convert list to numpy array
brand_vectors_array = np.array(brand_vectors_pandas['vector'].tolist())

# t-SNE 모델 설정 및 변환 수행
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=1004)
tsne_results = tsne.fit_transform(brand_vectors_array)

# t-SNE 결과를 pandas DataFrame에 추가
brand_vectors_pandas['tsne-2d-one'] = tsne_results[:,0]
brand_vectors_pandas['tsne-2d-two'] = tsne_results[:,1]

# COMMAND ----------

# t-SNE 결과 시각화
a = ["red"]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=a,
    data=brand_vectors_pandas,
    legend="full",
    alpha=0.3
)
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 중카별로 색깔 다르게 표현하기

# COMMAND ----------

brand_vectors = spark.read.table("asac.brand_vectors_2_cat")

# COMMAND ----------

brand_vectors = brand_vectors.filter(col("cat2").isNotNull())

# word와 cat2 컬럼이 모두 동일한 행을 하나만 남기고 나머지 중복 행 삭제
brand_vectors = brand_vectors.dropDuplicates(["word", "cat2"])

brand_vectors.show()

# COMMAND ----------

from pyspark.sql.functions import count
ca2 = brand_vectors.groupBy("cat2").count()
ca2.show()

# COMMAND ----------

model = Word2VecModel.load("dbfs:/FileStore/amazon/model/brand2vec_ver2")

# Word2Vec 모델로부터 브랜드 벡터 추출
# brand_vectors = model.getVectors()  # 카테고리 추가한거 위에서 불러왔기 때문에 각주처리
# Spark DataFrame을 pandas DataFrame으로 변환
brand_vectors_pandas = brand_vectors.toPandas()

import numpy as np


# Convert list to numpy array
brand_vectors_array = np.array(brand_vectors_pandas['vector'].tolist())

# t-SNE 모델 설정 및 변환 수행
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=1004)
tsne_results = tsne.fit_transform(brand_vectors_array)

# t-SNE 결과를 pandas DataFrame에 추가
brand_vectors_pandas['tsne-2d-one'] = tsne_results[:,0]
brand_vectors_pandas['tsne-2d-two'] = tsne_results[:,1]

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.color_palette("tab10", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=1
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.color_palette("hls", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=1
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.color_palette("pastel", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=1
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.color_palette("muted", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=1
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.color_palette("bright", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=1
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### brand 입력하면 코사인 높은거 5개, 낮은거 5개

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
brand_vectors = spark.read.table("asac.brand2vec_emb_2")
brand_vectors_pandas = brand_vectors.toPandas()

import numpy as np


# Convert list to numpy array
brand_vectors_array = np.array(brand_vectors_pandas['vector'].tolist())

# t-SNE 모델 설정 및 변환 수행
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=1004)
tsne_results = tsne.fit_transform(brand_vectors_array)

# t-SNE 결과를 pandas DataFrame에 추가
brand_vectors_pandas['tsne-2d-one'] = tsne_results[:,0]
brand_vectors_pandas['tsne-2d-two'] = tsne_results[:,1]

# COMMAND ----------

df_cos = spark.read.table("asac.brand2vec_emb_2_cos_3")

# COMMAND ----------

def related_brands(brand_1_value):
    # brand_1 값에 대해 em_cosine_similarity가 상위 5개
    related_brands = df_cos.filter(df_cos.brand_1 == brand_1_value) \
                           .orderBy(F.desc("em_cosine_similarity")) \
                           .limit(5) \
                           .select("brand_2") \
                           .collect()
    
    # 결과 반환
    if related_brands:
        return [row["brand_2"] for row in related_brands]
    else:
        return "해당 brand_1에 대한 데이터가 없습니다."

brand_1_input = "APPLE"
print(related_brands(brand_1_input))


# COMMAND ----------

def not_related_brand(brand_1_value):
    # brand_1 값에 대해 em_cosine_similarity가 하위 5개
    not_related_brand = df_cos.filter(df_cos.brand_1 == brand_1_value) \
                           .orderBy(F.asc("em_cosine_similarity")) \
                           .limit(5) \
                           .select("brand_2") \
                           .collect()
    
    # 결과 반환
    if not_related_brand:
        return [row["brand_2"] for row in not_related_brand]
    else:
        return "해당 brand_1에 대한 데이터가 없습니다."

brand_1_input = "APPLE"
print(not_related_brand(brand_1_input))

# COMMAND ----------

import matplotlib.pyplot as plt

# related_brands 함수를 수정했다고 가정, 여러 관련 브랜드를 리스트로 반환
input_brand = "APPLE"
related_brands_list = related_brands(input_brand)  # 상위 10개 관련 브랜드 리스트
non_related_brands_list = not_related_brand(input_brand)

# 관련 브랜드를 포함한 브랜드 벡터 필터링
filtered_brand_vectors = brand_vectors_pandas[brand_vectors_pandas['word'].isin([input_brand] + related_brands_list)]
plt.figure(figsize=(8,8))
# 입력 브랜드 시각화
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-two'],
    c='blue',  
    s=200,  
    alpha=0.5,
    label=input_brand  
)

# 관련 브랜드 시각화
for related_brand in related_brands_list:
    plt.scatter(
        x=filtered_brand_vectors[filtered_brand_vectors['word'] == related_brand]['tsne-2d-one'], 
        y=filtered_brand_vectors[filtered_brand_vectors['word'] == related_brand]['tsne-2d-two'],
        s=200,  
        alpha=0.5,
        label=related_brand
    )

# 범례 표시
plt.legend()

# 그래프 표시
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
# 가정된 상위 5개 관련 브랜드와 하위 5개 관련없는 브랜드 리스트
input_brand = "APPLE"
related_brands_list = related_brands(input_brand)  # 상위 10개 관련 브랜드 리스트
non_related_brands_list = not_related_brand(input_brand) # 하위 10개 관련없는 브랜드 리스트

# 관련 브랜드와 관련없는 브랜드를 포함한 브랜드 벡터 필터링
filtered_brand_vectors = brand_vectors_pandas[brand_vectors_pandas['word'].isin([input_brand] + related_brands_list + non_related_brands_list)]

# 입력 브랜드 시각화
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-two'],
    c='blue',  
    s=200,  
    alpha=0.5,
    label=input_brand  
)

# 관련 브랜드 시각화
for related_brand in related_brands_list:
    plt.scatter(
        x=filtered_brand_vectors[filtered_brand_vectors['word'] == related_brand]['tsne-2d-one'], 
        y=filtered_brand_vectors[filtered_brand_vectors['word'] == related_brand]['tsne-2d-two'],
        s=200,  
        alpha=0.5,
        label=related_brand
    )

# 관련없는 브랜드 시각화, 모양을 'x'로 지정
for non_related_brand in non_related_brands_list:
    plt.scatter(
        x=filtered_brand_vectors[filtered_brand_vectors['word'] == non_related_brand]['tsne-2d-one'], 
        y=filtered_brand_vectors[filtered_brand_vectors['word'] == non_related_brand]['tsne-2d-two'],
        s=200,  
        alpha=0.5,
        marker='*',  
        label=non_related_brand
    )

# 범례 표시
plt.legend()

# 그래프 표시
plt.show()


# COMMAND ----------


