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

amazon_data = spark.sql("""
select reviewTime, reviewerID, asin
from asac.review_cellphone_accessories_final
""")


# COMMAND ----------

amazon_data.show()

# COMMAND ----------

brand_data = spark.sql("""
select asin, brand
from asac.meta_cell_phones_and_accessories_new_price2
""")

# COMMAND ----------

brand_data.show()

# COMMAND ----------

amazon_data = amazon_data.join(brand_data, "asin", "inner")

# COMMAND ----------

amazon_data.show()

# COMMAND ----------

name = "asac.brand2vec"
amazon_data.write.saveAsTable(name)

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
# MAGIC ### 브랜드투벡 모델

# COMMAND ----------

word2Vec = Word2Vec(vectorSize=30, seed=1004, inputCol="sorted_list", outputCol="model", minCount=10, maxIter = 30, numPartitions = 16, windowSize = 5) 
model = word2Vec.fit(grouped_df)

model_path = "dbfs:/FileStore/amazon/model/brand2vec"
model.write().overwrite().save(model_path)

# COMMAND ----------

model = Word2VecModel.load("dbfs:/FileStore/amazon/model/brand2vec")

# COMMAND ----------

from pyspark.sql.functions import format_number as fmt
display(model.findSynonyms("Motorola", 10).select("word", fmt("similarity", 5).alias("similarity")))

# COMMAND ----------



# COMMAND ----------

brand_vec = model.getVectors()
brand_vec.createOrReplaceTempView("brand_vec")
display(brand_vec)

# COMMAND ----------

name = "asac.brand2vec_emb"
brand_vec.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 조인전 데이터

# COMMAND ----------

brand_vec = spark.read.table("asac.brand2vec_emb")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb
# MAGIC order by word

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand2vec_emb

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 조인 테이블 생성

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

display(df_filtered)

# COMMAND ----------

name = "asac.brand2vec_emb_cos_1"
df_filtered.write.saveAsTable(name)

# COMMAND ----------

df_filtered = spark.read.table("asac.brand2vec_emb_cos_1")

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

name = "asac.brand2vec_emb_cos_2"
df_filtered.write.saveAsTable(name)

# COMMAND ----------

df_filtered = spark.read.table("asac.brand2vec_emb_cos_2")

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

df_cos.show(1)

# COMMAND ----------

name = "asac.brand2vec_emb_cos_3"
df_cos.write.saveAsTable(name)

# COMMAND ----------

df_cos = spark.read.table("asac.brand2vec_emb_cos_3")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb_cos_3
# MAGIC where 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb_cos_3

# COMMAND ----------

# MAGIC %md
# MAGIC ### 중카 붙이기

# COMMAND ----------

cat2_data = spark.sql("""
select brand, cat2
from asac.meta_cell_phones_and_accessories_new_price2
""")

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,cat2  from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where brand == "3"

# COMMAND ----------

brand_vectors = spark.read.table("asac.brand2vec_emb")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand2vec_emb

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand2vec_emb

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct word) from asac.brand2vec_emb

# COMMAND ----------

cat2_data = cat2_data.withColumnRenamed("brand", "word")
brand_vectors_cat = brand_vectors.join(cat2_data, on="word", how="left")


# COMMAND ----------

brand_vectors_cat.show(1)

# COMMAND ----------

name = "asac.brand_vectors_cat"
brand_vectors_cat.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand_vectors_cat

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand_vectors_cat

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### t-sne 구현하기
# MAGIC - https://www.databricks.com/notebooks/reprisk_notebooks/02_rep_eda.html

# COMMAND ----------

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
brand_vectors = spark.read.table("asac.brand2vec_emb")
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

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hsv", 10),
    data=brand_vectors_pandas,
    legend="full",
    alpha=0.3
)
plt.show()

# COMMAND ----------

# t-SNE 결과 시각화

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("muted"),  
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

brand_vectors = spark.read.table("asac.brand_vectors_cat")

# COMMAND ----------

brand_vectors = brand_vectors.filter(col("cat2").isNotNull())

# word와 cat2 컬럼이 모두 동일한 행을 하나만 남기고 나머지 중복 행 삭제
brand_vectors = brand_vectors.dropDuplicates(["word", "cat2"])

brand_vectors.show()

# COMMAND ----------

brand_vectors.count()

# COMMAND ----------

model = Word2VecModel.load("dbfs:/FileStore/amazon/model/brand2vec")

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

palette = sns.color_palette("hsv", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=0.3
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
    alpha=0.3
)

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
    alpha=0.3
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat2'].nunique()

palette = sns.light_palette("blue", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat2",
    data=brand_vectors_pandas,
    legend="full",
    alpha=0.3
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 소카별로도 확인

# COMMAND ----------

cat3_data = spark.sql("""
select brand, cat3
from asac.meta_cell_phones_and_accessories_new_price2
""")

# COMMAND ----------

brand_vectors = spark.read.table("asac.brand2vec_emb")

# COMMAND ----------

cat3_data = cat3_data.withColumnRenamed("brand", "word")
brand_vectors_cat_3 = brand_vectors.join(cat3_data, on="word", how="left")

# COMMAND ----------

brand_vectors_cat_3.show(1)

# COMMAND ----------

name = "asac.brand_vectors_cat_3"
brand_vectors_cat_3.write.saveAsTable(name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.brand_vectors_cat_3

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.brand_vectors_cat_3

# COMMAND ----------

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# COMMAND ----------

brand_vectors = spark.read.table("asac.brand_vectors_cat_3")

# COMMAND ----------

brand_vectors = brand_vectors.filter(col("cat3").isNotNull())

# word와 cat3 컬럼이 모두 동일한 행을 하나만 남기고 나머지 중복 행 삭제
brand_vectors = brand_vectors.dropDuplicates(["word", "cat3"])

brand_vectors.show()

# COMMAND ----------

brand_vectors.count()

# COMMAND ----------

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

num_categories = brand_vectors_pandas['cat3'].nunique()

palette = sns.color_palette("pastel", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat3",
    data=brand_vectors_pandas,
    alpha=0.3,
    legend=False
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat3'].nunique()

palette = sns.color_palette("tab10", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat3",
    data=brand_vectors_pandas,
    alpha=0.3,
    legend=False
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### cat3 1000번이상 등장한  행만 저장

# COMMAND ----------

from pyspark.sql import functions as F

# `cat3` 기준으로 그룹화하여 등장 횟수(count) 계산
cat3_counts = brand_vectors.groupBy("cat3").count()

# 등장 횟수가 500번 이상인 `cat3` 값만 필터링
cat3_filtered = cat3_counts.filter(cat3_counts["count"] >= 1000)

# 필터링된 `cat3` 값들의 리스트 생성
cat3_list = cat3_filtered.select("cat3").rdd.flatMap(lambda x: x).collect()

# 원본 DataFrame에서 필터링된 `cat3` 값들을 가진 행만 선택
brand_vectors = brand_vectors.filter(brand_vectors["cat3"].isin(cat3_list))


# COMMAND ----------

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

num_categories = brand_vectors_pandas['cat3'].nunique()

palette = sns.color_palette("tab10", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat3",
    data=brand_vectors_pandas,
    alpha=0.3,
    legend=False
)

# COMMAND ----------

num_categories = brand_vectors_pandas['cat3'].nunique()

palette = sns.color_palette("pastel", num_categories)

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=palette,
    hue="cat3",
    data=brand_vectors_pandas,
    alpha=0.3,
    legend=False
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 임의로 두 개 지정해서 확인

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
brand_vectors = spark.read.table("asac.brand2vec_emb")
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


# 'Knisay'와 'AMIR' 브랜드만 필터링
filtered_brand_vectors = brand_vectors_pandas[brand_vectors_pandas['word'].isin(['Knisay', 'AMIR'])]

# t-SNE 결과 시각화
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="word",  # 브랜드 이름에 따라 색깔을 구분합니다.
    palette=sns.color_palette("hsv", 2),  # 'Knisay'와 'AMIR' 두 브랜드만 표시하므로 색상 팔레트 크기를 2로 설정합니다.
    data=filtered_brand_vectors,
    legend="full",
    alpha=0.3
)
plt.show()


# COMMAND ----------

plt.figure(figsize=(16,10))
plt.scatter(
    x=filtered_brand_vectors['tsne-2d-one'], 
    y=filtered_brand_vectors['tsne-2d-two'],
    c=['blue' if brand == 'Knisay' else 'red' for brand in filtered_brand_vectors['word']],  # Knisay는 파란색, AMIR는 빨간색으로 표시
    s=200,  # 점의 크기 설정
    alpha=0.3
)

plt.legend(['Knisay', 'AMIR'])
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Knisay에 대한 데이터 포인트를 그리기
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == 'Knisay']['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == 'Knisay']['tsne-2d-two'],
    c='blue',  
    s=200,  
    alpha=0.3,
    label='Knisay'  # Knisay에 대한 라벨 추가
)

# AMIR에 대한 데이터 포인트를 그리기
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == 'AMIR']['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == 'AMIR']['tsne-2d-two'],
    c='red',  
    s=200,  
    alpha=0.3,
    label='AMIR'  # AMIR에 대한 라벨 추가
)

# 범례를 표시
plt.legend()

plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### brand 입력하면, 코사인 유사도 가장 높은 브랜드 뽑아서 그래프 그리기

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
brand_vectors = spark.read.table("asac.brand2vec_emb")
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

df_cos = spark.read.table("asac.brand2vec_emb_cos_3")

# COMMAND ----------

def related_brand(brand_1_value):
    # brand_1 값에 대해 em_cosine_similarity가 가장 높은 행 필터링
    related_brand = df_cos.filter(df_cos.brand_1 == brand_1_value) \
                              .orderBy(F.desc("em_cosine_similarity")) \
                              .limit(1) \
                              .select("brand_2") \
                              .collect()
    
    # 결과 반환
    if related_brand:
        return related_brand[0]["brand_2"]
    else:
        return "해당 brand_1에 대한 데이터가 없습니다."

brand_1_input = "APPLE"
print(related_brand(brand_1_input))

# COMMAND ----------

def not_related_brand(brand_1_value):
    # brand_1 값에 대해 em_cosine_similarity가 가장낮은 행 필터링
    not_related_brand = df_cos.filter(df_cos.brand_1 == brand_1_value) \
                              .orderBy(F.asc("em_cosine_similarity")) \
                              .limit(1) \
                              .select("brand_2") \
                              .collect()
    
    # 결과 반환
    if not_related_brand:
        return not_related_brand[0]["brand_2"]
    else:
        return "해당 brand_1에 대한 데이터가 없습니다."

brand_1_input = "3M"
print(not_related_brand(brand_1_input))

# COMMAND ----------

import matplotlib.pyplot as plt

input_brand = brand_1_input
related = related_brand(input_brand)
not_related = not_related_brand(input_brand)
filtered_brand_vectors = brand_vectors_pandas[brand_vectors_pandas['word'].isin([input_brand, related,not_related])]

# 입력 브랜드
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == input_brand]['tsne-2d-two'],
    c='blue',  
    s=200,  
    alpha=0.3,
    label=input_brand  
)

# 출력 브랜드
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == related]['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == related]['tsne-2d-two'],
    c='red',  
    s=200,  
    alpha=0.3,
    label=related 
)

# 출력 브랜드
plt.scatter(
    x=filtered_brand_vectors[filtered_brand_vectors['word'] == not_related]['tsne-2d-one'], 
    y=filtered_brand_vectors[filtered_brand_vectors['word'] == not_related]['tsne-2d-two'],
    c='black',  
    s=200,  
    alpha=0.3,
    label=not_related 
)


plt.legend()
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


