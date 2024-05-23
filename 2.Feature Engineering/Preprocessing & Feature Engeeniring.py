#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Co_review Data set 생성
# --> 아이템 간 연간관계 나타내는 데이터

coreview = spark.sql('''
SELECT *
FROM (
    SELECT 
        a.asin AS asin1, 
        b.asin AS asin2,
        COUNT(DISTINCT a.reviewerID) review_cnts,
        COUNT(DISTINCT CASE WHEN a.overall = 5 AND b.overall = 5 THEN a.reviewerID ELSE NULL END) review_5_cnts,
        COUNT(DISTINCT CASE WHEN a.overall >= 4 AND b.overall >= 4 THEN a.reviewerID ELSE NULL END) review_4more_cnts,
        COUNT(DISTINCT CASE WHEN a.overall >= 3 AND b.overall >= 3 THEN a.reviewerID ELSE NULL END) review_3more_cnts
    FROM asac.trial_cell AS a
    INNER JOIN asac.trial_cell AS b
    ON a.reviewerID = b.reviewerID
    AND YEAR(a.reviewTime) = YEAR(b.reviewTime)
    AND MONTH(a.reviewTime) = MONTH(b.reviewTime)
    GROUP BY a.asin, b.asin
) AS t
WHERE t.asin1 != t.asin2
''')

coreview.write.format("delta").saveAsTable("asac.new_co_review_cellphones_and_accessories_same_year_and_month")


# In[ ]:


# 코리뷰 & 메타데이터 쪼인을 통해 변수 추가

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# SparkSession 초기화
spark = SparkSession.builder.appName("Cellphone Data Join").getOrCreate()

# asin1을 기준으로 조인
joined_df_asin1 = cellphone_coreview.join(
    cellphone_meta,
    cellphone_coreview.asin1 == cellphone_meta.asin,
    'left'
).select(
    cellphone_coreview["*"],  # 기존 cellphone_coreview의 모든 칼럼
    col("cat2").alias("asin1_cat2"),
    col("cat3").alias("asin1_cat3"),
    col("new_price2").alias("asin1_price")
)

# asin2를 기준으로 조인
joined_df_asin2 = cellphone_coreview.join(
    cellphone_meta,
    cellphone_coreview.asin2 == cellphone_meta.asin,
    'left'
).select(
    cellphone_coreview["*"],  # 기존 cellphone_coreview의 모든 칼럼
    col("cat2").alias("asin2_cat2"),
    col("cat3").alias("asin2_cat3"),
    col("new_price2").alias("asin2_price")
)

# 조인된 결과를 합치기 위해 필요한 칼럼만 선택
joined_df = joined_df_asin1.join(
    joined_df_asin2,
    ["Year", "Month", "asin1", "asin2", "review_cnts"],  # 'Year'와 'Month'를 공통 칼럼에 추가
    "inner"
).select(
    "Year",  
    "Month",
    "asin1",
    "asin2",
    "review_cnts",
    "asin1_cat2",
    "asin1_cat3",
    "asin1_price",
    "asin2_cat2",
    "asin2_cat3",
    "asin2_price"
)

# 결과 확인
display(joined_df)


# In[ ]:


# 전처리

from pyspark.sql import functions as F

# Category NULL 값 unknown 으로 변경
joined_df = joined_df.withColumn("asin1_cat2", F.when(F.col("asin1_cat2").isNull(), "unknown").otherwise(F.col("asin1_cat2")))
joined_df = joined_df.withColumn("asin1_cat3", F.when(F.col("asin1_cat3").isNull(), "unknown").otherwise(F.col("asin1_cat3")))
joined_df = joined_df.withColumn("asin2_cat2", F.when(F.col("asin2_cat2").isNull(), "unknown").otherwise(F.col("asin2_cat2")))
joined_df = joined_df.withColumn("asin2_cat3", F.when(F.col("asin2_cat3").isNull(), "unknown").otherwise(F.col("asin2_cat3")))

# Price null 값 -999로 변경
joined_df = joined_df.withColumn("asin1_price", F.when(F.col("asin1_price").isNull(), -999).otherwise(F.col("asin1_price")))
joined_df = joined_df.withColumn("asin2_price", F.when(F.col("asin2_price").isNull(), -999).otherwise(F.col("asin2_price")))


from pyspark.sql import functions as F

# asin1 중카테고리 == asin2 중카테고리
joined_df = joined_df.withColumn("cat2_same", F.when(F.col("asin1_cat2") == F.col("asin2_cat2"), 1).otherwise(0))

# asin1 소카테고리 == asin2 소카테고리
joined_df = joined_df.withColumn("cat3_same", F.when(F.col("asin1_cat3") == F.col("asin2_cat3"), 1).otherwise(0))

# asin1 중&소카테고리 == asin2 중&소카테고리
joined_df = joined_df.withColumn("cat2_and_cat3_same", F.when((F.col("asin1_cat2") == F.col("asin2_cat2")) & 
                                                                (F.col("asin1_cat3") == F.col("asin2_cat3")), 1).otherwise(0))

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 이전 연도와 이전 달 계산
joined_df = joined_df.withColumn("prev_Year", F.when(F.col("Month") == 1, F.col("Year") - 1).otherwise(F.col("Year")))
joined_df = joined_df.withColumn("prev_Month", F.when(F.col("Month") == 1, 12).otherwise(F.col("Month") - 1))

# 조인을 위한 임시 키 생성
joined_df = joined_df.withColumn("joinKey", F.concat_ws("_", F.col("prev_Year"), F.col("prev_Month"), F.col("asin1")))
joined_df = joined_df.withColumn("joinKey2", F.concat_ws("_", F.col("prev_Year"), F.col("prev_Month"), F.col("asin2")))

# Window Specification 정의
windowSpec1 = Window.partitionBy("joinKey")
windowSpec2 = Window.partitionBy("joinKey2")

# 이전 달 등장 횟수 계산
joined_df = joined_df.withColumn("asin1_count_prevMonth", F.count("asin1").over(windowSpec1))
joined_df = joined_df.withColumn("asin2_count_prevMonth", F.count("asin2").over(windowSpec2))

# null 값을 0으로 대체
joined_df = joined_df.withColumn("asin1_count_prevMonth", F.coalesce(F.col("asin1_count_prevMonth"), F.lit(0)))
joined_df = joined_df.withColumn("asin2_count_prevMonth", F.coalesce(F.col("asin2_count_prevMonth"), F.lit(0)))

# 필요 없는 컬럼 제거
joined_df = joined_df.drop("joinKey", "joinKey2","prev_Month","prev_Year")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 문자열 칼럼들 지정
categoricalColumns = ["asin1_cat2", "asin1_cat3", "asin2_cat2", "asin2_cat3"]

# 각 문자열 칼럼에 대해 StringIndexer 및 OneHotEncoder 적용
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(joined_df) for column in categoricalColumns]
encoders = [OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[indexer.getOutputCol().replace("_index", "_vec")]) for indexer in indexers]

# 가격 유사도 계산 함수
def price_similarity(price1, price2):
    return (2 * price1 * price2) / (price1 + price2) if (price1 + price2) != 0 else 0

# UDF로 등록
price_similarity_udf = udf(price_similarity, DoubleType())

# Pipeline 생성 및 적용
pipeline = Pipeline(stages=indexers + encoders)
joined_df = pipeline.fit(joined_df).transform(joined_df)

# 가격 유사도 칼럼 추가
joined_df = joined_df.withColumn("price_similarity", price_similarity_udf(col("asin1_price"), col("asin2_price")))

# 원래 문자열 칼럼 및 기존 가격 칼럼 삭제
for column in categoricalColumns + ["asin1_price", "asin2_price"]:
    joined_df = joined_df.drop(column)

# 인덱스 칼럼을 삭제
for column in ["asin1_cat2_index", "asin1_cat3_index", "asin2_cat2_index", "asin2_cat3_index"]:
    joined_df = joined_df.drop(column)

display(joined_df)


# In[ ]:


# Train & Test Split

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

test_df = joined_df.filter((col("Year") == 2017) & (col("Month") >= 10) | (col("Year") == 2018) & (col("Month") <= 10))
train_df = joined_df.subtract(test_df)


# In[ ]:


# Train Data Labeling
# --> 데이터 충분성 고려하여 3 기준으로 라벨링 실행

from pyspark.sql.functions import when


# 1) review_cnts 값이 3 이상인 경우 1, 그렇지 않은 경우 0
train_df = train_df.withColumn("target", when(train_df.review_cnts >= 3, 1).otherwise(0))

display(train_df)


# In[ ]:


# Coreview Data 내 상품조합 중복 데이터 삭제
# --> ab=ba 삭제

from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, lit, sort_array, min, struct
from pyspark.sql.window import Window
# asin1과 asin2를 정렬하여 새로운 칼럼에 저장
train_df_abba_delete = train_df.withColumn("sorted_asins", sort_array(array("asin1", "asin2")))

# Window 정의 - sorted_asins, Year, Month 기준으로 정렬
windowSpec = Window.partitionBy("sorted_asins", "Year", "Month").orderBy("asin1", "asin2")

# 각 그룹 내에서 먼저 등장하는 행에 대한 표시
train_df_abba_delete = train_df_abba_delete.withColumn("row", min(struct("asin1", "asin2")).over(windowSpec))

# 먼저 등장하는 행만 필터링
df_filtered = train_df_abba_delete.filter(col("row.asin1") == col("asin1")) \
                .drop("sorted_asins", "row")

display(df_filtered)


# In[ ]:




