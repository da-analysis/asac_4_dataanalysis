# Databricks notebook source
# MAGIC %md
# MAGIC ### 임베딩, 통계량 , 코사인 유사도 데이터 결합 및 계산
# MAGIC - 32차원으로 결정

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

# 임시 데이터 프레임 생성
# 데이터 프레임 정해지면, 이건 안해도됨

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

# COMMAND ----------

display(df_final_22)

# COMMAND ----------

# asin1과 asin2 를 기준으로 join 진행
# 임베딩, 통계량 값, 길이

df_final_22 = df_final_22.join(cell_32, df_final_22.asin1 == cell_32.asin)
df_final_22 = df_final_22.drop("asin")
for col_name in df_final_22.columns:
    if col_name not in ["asin1", "asin2"]:  # asin1과 asin2를 제외
        df_final_22 = df_final_22.withColumnRenamed(col_name, col_name + "_1")

cell_32_renamed = cell_32.select(['asin'] + [col(c).alias(c + '_2') for c in cell_32.columns if c != 'asin'])

# df_final_22와 변경된 cell_32_renamed 조인
df_final_22 = df_final_22.join(cell_32_renamed, df_final_22.asin2 == cell_32_renamed.asin)

# 필요하지 않은 cell_32의 asin 컬럼 삭제
df_final_22 = df_final_22.drop(cell_32_renamed.asin)

df_final_22 = df_final_22.drop("asin")
df_final_22 = df_final_22.drop("variance_1")
df_final_22 = df_final_22.drop("variance_2")

# COMMAND ----------

df_final_33 = df_final_22
df_final_44 = df_final_22
df_final_55 = df_final_22

# COMMAND ----------

display(df_final_22)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 방법 1) udf 쓴거 (2분)

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
    df_final_33 = df_final_33.withColumn(f"{column_name}_cosine_similarity", 
                                   cosine_similarity_udf(col(f"{column_name}_1"), col(f"{column_name}_2")))

# 코사인 유사도 컬럼들의 평균을 계산하는 UDF
average_udf = udf(lambda arr: float(sum(arr)) / len(arr), DoubleType())

# 코사인 유사도 컬럼들의 리스트
columns_to_average = [f"{column_name}_cosine_similarity" for column_name in columns]

# 평균 코사인 유사도 컬럼 추가
df_final_33 = df_final_33.withColumn("average_cosine_similarity", average_udf(array(*columns_to_average)))
display(df_final_33)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 방법2) udf 안쓴거 (4분)

# COMMAND ----------

from pyspark.sql.functions import col, sqrt, sum as _sum, when
columns = [
    ("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"),("q1_1", "q1_2"),
    ("q2_1", "q2_2"),("q3_1", "q3_2"),("skewness_1", "skewness_2"),("kurtosis_1", "kurtosis_2")
]

for col1,col2 in columns:
    df_final_44 = df_final_44.withColumn("dot_product", sum(col(col1).getItem(i) * col(col2).getItem(i) for i in range(32)))
    df_final_44 = df_final_44.withColumn("norm_v1", sqrt(sum(col(col1).getItem(i) ** 2 for i in range(32))))
    df_final_44 = df_final_44.withColumn("norm_v2", sqrt(sum(col(col2).getItem(i) ** 2 for i in range(32))))
    df_final_44 = df_final_44.withColumn(f"{col1[:-2]}_cosine_similarity", col("dot_product") / (col("norm_v1") * col("norm_v2")))
    df_final_44 = df_final_44.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])
    df_final_44 = df_final_44.drop("dot_product", "norm_v1", "norm_v2")


df_final_44 = df_final_44.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

display(df_final_44)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 방법3) 더 효율적으로 하는 방법(2분)

# COMMAND ----------

from pyspark.sql.functions import expr, col
from pyspark.sql.functions import col, sqrt, sum as _sum, when
columns = [
    ("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"),("q1_1", "q1_2"),
    ("q2_1", "q2_2"),("q3_1", "q3_2"),("skewness_1", "skewness_2"),("kurtosis_1", "kurtosis_2")
]

# 각 컬럼 쌍에 대해 반복
for col1, col2 in columns:
    # Dot product
    dot_product_expr = " + ".join([f"({col1}[{i}]) * ({col2}[{i}])" for i in range(32)])
    
    # Norms
    norm_v1_expr = "SQRT(" + " + ".join([f"({col1}[{i}]) * ({col1}[{i}])" for i in range(32)]) + ")"
    norm_v2_expr = "SQRT(" + " + ".join([f"({col2}[{i}]) * ({col2}[{i}])" for i in range(32)]) + ")"
    
    # Cosine Similarity
    cosine_similarity_expr = f"({dot_product_expr}) / ({norm_v1_expr} * {norm_v2_expr})"
    
    # DataFrame에 코사인 유사도 컬럼 추가
    df_final_55 = df_final_55.withColumn(f"{col1[:-2]}_cosine_similarity", expr(cosine_similarity_expr))
    df_final_55 = df_final_55.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])

# 최종 코사인 유사도 평균 계산
df_final_55 = df_final_55.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

# COMMAND ----------

display(df_final_55)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 방법4) pandas udf 활용 (1분)

# COMMAND ----------

df_final_66 = df_final_22

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import numpy as np

columns = [("mean_vector_1", "mean_vector_2"), ("std_dev_1", "std_dev_2"),("q1_1", "q1_2"),
    ("q2_1", "q2_2"),("q3_1", "q3_2"),("skewness_1", "skewness_2"),("kurtosis_1", "kurtosis_2")]

@pandas_udf("double", PandasUDFType.SCALAR)
def cosine_similarity_udf(v1: pd.Series, v2: pd.Series) -> pd.Series:
    # 각 Series의 요소가 벡터인 경우를 처리하기 위한 수정
    dot_product = np.array([np.dot(a, b) for a, b in zip(v1, v2)])
    norm_v1 = np.sqrt(np.array([np.dot(a, a) for a in v1]))
    norm_v2 = np.sqrt(np.array([np.dot(b, b) for b in v2]))
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return pd.Series(cosine_similarity)


# DataFrame에 코사인 유사도 컬럼 추가
for col1,col2 in columns:
    df_final_66 = df_final_66.withColumn(f"{col1[:-2]}_cosine_similarity",  cosine_similarity_udf(col(col1), col(col2)))
    df_final_66 = df_final_66.fillna(0, subset=[f"{col1[:-2]}_cosine_similarity"])


# 최종 코사인 유사도 평균 계산
df_final_66 = df_final_66.withColumn("cosine_fin", (col("mean_vector_cosine_similarity") + col("std_dev_cosine_similarity") + col("q1_cosine_similarity")
                                                    +col("q2_cosine_similarity")+col("q3_cosine_similarity")+col("skewness_cosine_similarity")
                                                    +col("kurtosis_cosine_similarity")) / 7)

display(df_final_66)  

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

## 여기서부턴 오류발생하는 것들

# COMMAND ----------

# udf 안쓰고 코사인 유사도 구하는 것
from pyspark.sql.functions import expr, col

columns = ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]
# 각 컬럼 쌍에 대해서 코사인 유사도를 계산
for column_name in columns:
    df_final_22 = df_final_22.withColumn(f"{column_name}_cosine_similarity",
    expr(f"""
    CASE 
        WHEN (sqrt(aggregate(transform({column_name}_1, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({column_name}_2, x -> x * x), 0.0, (acc, x) -> acc + x))) = 0 THEN 0
        ELSE (aggregate(transform(arrays_zip({column_name}_1, {column_name}_2), x -> x.col1 * x.col2), 0.0, (acc, x) -> acc + x)) / 
             (sqrt(aggregate(transform({column_name}_1, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({column_name}_2, x -> x * x), 0.0, (acc, x) -> acc + x)))
    END
    """))

# 코사인 유사도 컬럼들의 리스트를 만들고 평균 코사인 유사도 계산
cosine_similarity_columns = [f"{column_name}_cosine_similarity" for column_name in columns]
expr_for_average = " + ".join(cosine_similarity_columns) + " / " + str(len(cosine_similarity_columns))

df_final_22 = df_final_22.withColumn("average_cosine_similarity", expr(expr_for_average))

# COMMAND ----------

from pyspark.sql.functions import expr, col

# df_final_22 데이터프레임을 가정
columns = [
    ("mean_vector_1", "mean_vector_2"),
    ("std_dev_1", "std_dev_2"),
    ("q1_1", "q1_2"),
    ("q2_1", "q2_2"),
    ("q3_1", "q3_2"),
    ("skewness_1", "skewness_2"),
    ("kurtosis_1", "kurtosis_2")
]

for col1, col2 in columns:
    df_final_22 = df_final_22.withColumn(f"{col1[:-2]}_cosine_similarity", 
        expr(f"""
            CASE 
                WHEN (sqrt(aggregate(transform({col1}, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({col2}, x -> x * x), 0.0, (acc, x) -> acc + x))) = 0 THEN 0
                ELSE (aggregate(transform(arrays_zip({col1}, {col2}), x -> x.{col1} * x.{col2}), 0.0, (acc, x) -> acc + x)) / 
                     (sqrt(aggregate(transform({col1}, x -> x * x), 0.0, (acc, x) -> acc + x)) * sqrt(aggregate(transform({col2}, x -> x * x), 0.0, (acc, x) -> acc + x)))
            END
        """)
    )

# 코사인 유사도 컬럼들의 리스트를 만들기
cosine_similarity_columns = [f"{col1[:-2]}_cosine_similarity" for col1, col2 in columns]

# 평균 코사인 유사도 계산
expr_for_average = " + ".join(cosine_similarity_columns) + " / " + str(len(cosine_similarity_columns))
df_final_22 = df_final_22.withColumn("average_cosine_similarity", expr(expr_for_average))


# COMMAND ----------

from pyspark.sql.functions import expr

# columns 변수에는 비교할 벡터 컬럼 이름들이 저장되어 있습니다.
columns = ["mean_vector", "std_dev", "q1", "q2", "q3", "skewness", "kurtosis"]

# 각 컬럼 쌍에 대해서 코사인 유사도를 계산합니다.
for column_name in columns:
    dot_product_expr = f"aggregate(transform({column_name}_1, (x, i) -> x * {column_name}_2[i]), 0.0, (acc, x) -> acc + x)"
    norm_1_expr = f"sqrt(aggregate(transform({column_name}_1, x -> x * x), 0.0, (acc, x) -> acc + x))"
    norm_2_expr = f"sqrt(aggregate(transform({column_name}_2, x -> x * x), 0.0, (acc, x) -> acc + x))"
    
    # 코사인 유사도 계산식
    cosine_similarity_expr = f"""
    CASE 
        WHEN ({norm_1_expr} * {norm_2_expr}) = 0 THEN 0
        ELSE {dot_product_expr} / ({norm_1_expr} * {norm_2_expr})
    END
    """
    
    # 코사인 유사도 컬럼 추가
    df_final_22 = df_final_22.withColumn(f"{column_name}_cosine_similarity", expr(cosine_similarity_expr))

# 코사인 유사도 컬럼들의 리스트를 만들고 평균 코사인 유사도 계산
cosine_similarity_columns = [f"{column_name}_cosine_similarity" for column_name in columns]
expr_for_average = " + ".join(cosine_similarity_columns) + " / " + str(len(cosine_similarity_columns))

# DECIMAL(1,1)로 CAST하여 타입 맞추기
df_final_22 = df_final_22.withColumn("average_cosine_similarity", expr(expr_for_average).cast("decimal(38, 10)"))


# COMMAND ----------



# COMMAND ----------


