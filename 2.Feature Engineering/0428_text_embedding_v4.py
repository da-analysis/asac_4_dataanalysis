# Databricks notebook source
# MAGIC %md
# MAGIC ### 코사인 유사도 순위 32차원과 128차원 한번 더 확인해보기

# COMMAND ----------

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

# COMMAND ----------

cell_32 = spark.read.table("asac.embed_cell_sbert_32_fin")
cell_128 = spark.read.table("asac.embed_cell_sbert_128_fin")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 32차원 먼저
# MAGIC - 기준 : B009EVGAVE

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.functions import array
target_asin = "B009EVGAVE"
target_row = cell_32.filter(col("asin") == target_asin).select(*["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]).collect()[0]


def cosine_similarity(v1, v2):
    dot_product = float(v1.dot(v2))
    norm_v1 = float(v1.norm(2))
    norm_v2 = float(v2.norm(2))
    # 두 벡터 중 하나라도 0 벡터인 경우, 코사인 유사도가 정의되지 않으므로 0을 반환
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    else:
        return dot_product / (norm_v1 * norm_v2)
    
def create_cosine_similarity_udf(target_vector):
    def func(v):
        return cosine_similarity(Vectors.dense(v), target_vector)
    return udf(func, DoubleType())

# 각 열에 대한 코사인 유사도 계산 및 새로운 열 추가
for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]:
    target_vector = Vectors.dense(target_row[column_name])
    cosine_similarity_udf = create_cosine_similarity_udf(target_vector)
    cell_32 = cell_32.withColumn(f"{column_name}_cosine_similarity", 
                                 cosine_similarity_udf(col(column_name)))

columns_to_average = [f"{column_name}_cosine_similarity" for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]]
average_udf = udf(lambda arr: float(sum(arr)) / len(arr), DoubleType())

cell_32 = cell_32.withColumn("average_cosine_similarity", average_udf(array(*columns_to_average)))


display(cell_32)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# 코사인 유사도를 기준으로 내림차순 정렬
windowSpec = Window.orderBy(col("average_cosine_similarity").desc())
cell_32 = cell_32.withColumn("rank", row_number().over(windowSpec))

# COMMAND ----------

display(cell_32)

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = "B009EVGAVE"

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = "B00H71BXCY"

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = "B01GPLJ96S"

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = "B00HNKOOK2"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 128차원

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler

target_asin = "B009EVGAVE"
target_row = cell_128.filter(col("asin") == target_asin).select(
    *["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
).collect()[0]

def cosine_similarity(v1, v2):
    dot_product = float(v1.dot(v2))
    norm_v1 = float(v1.norm(2))
    norm_v2 = float(v2.norm(2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    else:
        return dot_product / (norm_v1 * norm_v2)

def create_cosine_similarity_udf(target_vector):
    def func(v):
        return cosine_similarity(Vectors.dense(v), target_vector)
    return udf(func, DoubleType())

# Looping over columns
for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]:
    target_vector = Vectors.dense(target_row[column_name])
    cosine_similarity_udf = create_cosine_similarity_udf(target_vector)
    cell_128 = cell_128.withColumn(
        f"{column_name}_cosine_similarity",
        cosine_similarity_udf(col(column_name))
    )

columns_to_average = [f"{column_name}_cosine_similarity" for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]]
average_udf = udf(lambda arr: float(sum(arr)) / len(arr), DoubleType())

cell_128 = cell_128.withColumn(
    "average_cosine_similarity",
    average_udf(array(*columns_to_average))
)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# 코사인 유사도를 기준으로 내림차순 정렬
windowSpec = Window.orderBy(col("average_cosine_similarity").desc())
cell_128 = cell_128.withColumn("rank", row_number().over(windowSpec))

display(cell_128.limit(100))

# COMMAND ----------

from pyspark.sql import functions as F

target_asin = "B009EVGAVE"
target_row = cell_128.filter(F.col("asin") == target_asin).select(
    *["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
).collect()[0]

# 각 컬럼에 대해 코사인 유사도를 계산
for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]:
    df_final = cell_128.withColumn(f"{column_name}_dot", F.expr(f"aggregate(arrays_zip({column_name}, array({', '.join([f'{target_row[column_name]}']*128)})), 0.0D, (acc, x) -> acc + (x.{column_name} * x.{target_row[column_name]}))")) \
        .withColumn(f"{column_name}_norm1", F.expr(f"sqrt(aggregate({column_name}, 0.0D, (acc, x) -> acc + (x * x)))")) \
        .withColumn(f"{column_name}_norm2", F.expr(f"sqrt(aggregate(array({', '.join([f'{target_row[column_name]}']*128)}), 0.0D, (acc, x) -> acc + (x * x)))")) \
        .withColumn(f"{column_name}_cosine_similarity", F.expr(f"{column_name}_dot / ({column_name}_norm1 * {column_name}_norm2)"))

# 코사인 유사도의 평균 계산
columns_to_average = [f"{column_name}_cosine_similarity" for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]]
df_final = df_final.withColumn(
    "average_cosine_similarity",
    F.expr("+".join(columns_to_average) + f"/ {len(columns_to_average)}")
)

# COMMAND ----------

from pyspark.sql import functions as F

target_asin = "B009EVGAVE"
target_row = cell_128.filter(F.col("asin") == target_asin).select(
    *["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
).collect()[0]

# 각 컬럼에 대해 코사인 유사도를 계산하는 반복문
for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]:
    df_final = cell_128.withColumn(
        f"{column_name}_dot", 
        F.expr(f"aggregate(arrays_zip({column_name}, array({', '.join([f'{target_row[column_name]}' for _ in range(128)])})), 0.0D, (acc, x) -> acc + (x.{column_name} * x['{target_row[column_name]}'])))")
    .withColumn(
        f"{column_name}_norm1", 
        F.expr(f"sqrt(aggregate(transform({column_name}, x -> x * x), 0.0D, (acc, x) -> acc + x)))")
    .withColumn(
        f"{column_name}_norm2", 
        F.expr(f"sqrt(aggregate(transform(array({', '.join([f'{target_row[column_name]}' for _ in range(128)])}), x -> x * x), 0.0D, (acc, x) -> acc + x)))")
    .withColumn(
        f"{column_name}_cosine_similarity", 
        F.expr(f"{column_name}_dot / ({column_name}_norm1 * {column_name}_norm2)")
    )

# 코사인 유사도의 평균을 계산
columns_to_average = [f"{column_name}_cosine_similarity" for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]]
df_final = df_final.withColumn(
    "average_cosine_similarity",
    F.expr("({}) / {}".format(" + ".join(columns_to_average), len(columns_to_average)))
)


# COMMAND ----------

from pyspark.sql import functions as F

target_asin = "B009EVGAVE"
target_row = cell_128.filter(F.col("asin") == target_asin).select(
    *["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
).collect()[0]

# 각 컬럼에 대해 코사인 유사도를 계산하는 반복문
for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]:
    df_final = cell_128.withColumn(
        f"{column_name}_dot", 
        F.expr(f"aggregate(arrays_zip({column_name}, array({', '.join([f'{target_row[column_name]}' for _ in range(128)])})), 0.0D, (acc, x) -> acc + (x.{column_name} * x['{target_row[column_name]}'])))")
    .withColumn(
        f"{column_name}_norm1", 
        F.expr(f"sqrt(aggregate(transform({column_name}, x -> x * x), 0.0D, (acc, x) -> acc + x)))")
    .withColumn(
        f"{column_name}_norm2", 
        F.expr(f"sqrt(aggregate(transform(array({', '.join([f'{target_row[column_name]}' for _ in range(128)])}), x -> x * x), 0.0D, (acc, x) -> acc + x)))")
    .withColumn(
        f"{column_name}_cosine_similarity", 
        F.expr(f"{column_name}_dot / ({column_name}_norm1 * {column_name}_norm2)")
    )

# 코사인 유사도의 평균을 계산
columns_to_average = [f"{column_name}_cosine_similarity" for column_name in ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]]
df_final = df_final.withColumn(
    "average_cosine_similarity",
    F.expr("({}) / {}".format(" + ".join(columns_to_average), len(columns_to_average)))
)


# COMMAND ----------

display(cell_128)

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = "B009EVGAVE"

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = ""

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = ""

# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final
# MAGIC where asin = ""

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select reviewText, asin from asac.review_cellphone_accessories_final

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## asin 맞춰서 값 옆으로 붙어오기
# MAGIC

# COMMAND ----------

display(cell_32)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
df = cell_32.select(["asin"])
df = df.withColumn("index", monotonically_increasing_id())

# 인덱스를 기준으로 1~10행과 11~20행 선택
df1 = df.filter((df["index"] < 10))
df2 = df.filter((df["index"] >= 10) & (df["index"] < 20))

# 인덱스 컬럼 제거
df1 = df1.drop("index")
df2 = df2.drop("index")

# 결과 확인
df1.show()
df2.show()

# COMMAND ----------

df1 = df1.withColumnRenamed("asin","asin1")
df2 = df2.withColumnRenamed("asin","asin2")


# COMMAND ----------

df1_with_index = df1.withColumn("index", monotonically_increasing_id())
df2_with_index = df2.withColumn("index", monotonically_increasing_id())

# 인덱스를 기준으로 조인하여 옆으로 결합
df_final = df1_with_index.join(df2_with_index, "index", "outer").drop("index")
df_final.show()

# COMMAND ----------

df_final.columns

# COMMAND ----------

# df_final과 cell_32 결합
# 'asin1' 컬럼과 'asin' 컬럼을 기준으로 join 수행
df_final = df_final.join(cell_32, df_final.asin1 == cell_32.asin)

# COMMAND ----------

df_final = df_final.drop("asin")

# COMMAND ----------

display(df_final)

# COMMAND ----------

for col_name in df_final.columns:
    if col_name not in ["asin1", "asin2"]:  # asin1과 asin2를 제외
        df_final = df_final.withColumnRenamed(col_name, col_name + "_1")

# COMMAND ----------

display(df_final)

# COMMAND ----------

cell_32_renamed = cell_32.select(['asin'] + [col(c).alias(c + '_2') for c in cell_32.columns if c != 'asin'])

# df_final과 변경된 cell_32_renamed 조인
df_final = df_final.join(cell_32_renamed, df_final.asin2 == cell_32_renamed.asin)

# 필요하지 않은 cell_32의 asin 컬럼 삭제
df_final = df_final.drop(cell_32_renamed.asin)

# COMMAND ----------

display(df_final)

# COMMAND ----------

df_final = df_final.drop("asin")

# COMMAND ----------

# 각각의 코사인 유사도 구하기
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, ArrayType

# 코사인 유사도를 계산하는 함수
def cosine_similarity(v1, v2):
    dot_product = float(v1.dot(v2))
    norm_v1 = float(v1.norm(2))
    norm_v2 = float(v2.norm(2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    else:
        return dot_product / (norm_v1 * norm_v2)
    
def create_cosine_similarity_udf():
    def func(v1, v2):
        return cosine_similarity(Vectors.dense(v1), Vectors.dense(v2))
    return udf(func, DoubleType())

cosine_similarity_udf = create_cosine_similarity_udf()

# 컬럼 쌍에 대해 코사인 유사도 계산 및 새 컬럼 추가
columns = ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
for column_name in columns:
    df_final = df_final.withColumn(f"{column_name}_cosine_similarity", 
                                   cosine_similarity_udf(col(f"{column_name}_1"), col(f"{column_name}_2")))

# 코사인 유사도 컬럼들의 평균을 계산하는 UDF
average_udf = udf(lambda arr: float(sum(arr)) / len(arr), DoubleType())

# 코사인 유사도 컬럼들의 리스트
columns_to_average = [f"{column_name}_cosine_similarity" for column_name in columns]

# 평균 코사인 유사도 컬럼 추가
df_final = df_final.withColumn("average_cosine_similarity", average_udf(array(*columns_to_average)))


# COMMAND ----------

display(df_final)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## udf 안쓰고 코사인유사도 구하기

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
df = cell_32.select(["asin"])
df = df.withColumn("index", monotonically_increasing_id())

# 인덱스를 기준으로 1~10행과 11~20행 선택
df1 = df.filter((df["index"] < 10))
df2 = df.filter((df["index"] >= 10) & (df["index"] < 20))

# 인덱스 컬럼 제거
df1 = df1.drop("index")
df2 = df2.drop("index")

df1 = df1.withColumnRenamed("asin","asin1")
df2 = df2.withColumnRenamed("asin","asin2")

df1_with_index = df1.withColumn("index", monotonically_increasing_id())
df2_with_index = df2.withColumn("index", monotonically_increasing_id())

# 인덱스를 기준으로 조인하여 옆으로 결합
df_final_22 = df1_with_index.join(df2_with_index, "index", "outer").drop("index")

df_final_22 = df_final_22.join(cell_32, df_final_22.asin1 == cell_32.asin)
df_final_22 = df_final_22.drop("asin")
for col_name in df_final_22.columns:
    if col_name not in ["asin1", "asin2"]:  # asin1과 asin2를 제외
        df_final_22 = df_final_22.withColumnRenamed(col_name, col_name + "_1")

cell_32_renamed = cell_32.select(['asin'] + [col(c).alias(c + '_2') for c in cell_32.columns if c != 'asin'])

# df_final과 변경된 cell_32_renamed 조인
df_final_22 = df_final_22.join(cell_32_renamed, df_final_22.asin2 == cell_32_renamed.asin)

# 필요하지 않은 cell_32의 asin 컬럼 삭제
df_final_22 = df_final_22.drop(cell_32_renamed.asin)

df_final_22 = df_final_22.drop("asin")

# COMMAND ----------

display(df_final_22)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col

# columns 변수에는 비교할 벡터 컬럼 이름들이 저장되어 있습니다.
columns = ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]

# 각 컬럼 쌍에 대해서 코사인 유사도를 계산합니다.
for column_name in columns:
    df_final_22 = df_final_22.withColumn(f"{column_name}_cosine_similarity",
        expr(f"""
        CASE 
            WHEN (sqrt(aggregate(transform({column_name}_1, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({column_name}_2, x -> x * x), 0.0, (acc, x) -> acc + x))) = 0 THEN 0
            ELSE (aggregate(transform(arrays_zip({column_name}_1, {column_name}_2), x -> x.`0` * x.`1`), 0.0, (acc, x) -> acc + x)) / 
                 (sqrt(aggregate(transform({column_name}_1, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({column_name}_2, x -> x * x), 0.0, (acc, x) -> acc + x)))
        END
        """))

# 코사인 유사도 컬럼들의 리스트를 만들고 평균 코사인 유사도를 계산합니다.
cosine_similarity_columns = [f"{column_name}_cosine_similarity" for column_name in columns]
expr_for_average = " + ".join(cosine_similarity_columns) + " / " + str(len(cosine_similarity_columns))

df_final_22 = df_final_22.withColumn("average_cosine_similarity", expr(expr_for_average))


# COMMAND ----------


