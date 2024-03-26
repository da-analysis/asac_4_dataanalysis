# Databricks notebook source
import pandas as pd
import json
import numpy as np
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4개의 파티션으로 나누어서 데이터 저장 및 불러오기

# COMMAND ----------

cell_phones_and_accessories = sc.textFile("dbfs:/FileStore/amazon/metadata/meta_Cell_Phones_and_Accessories.json")

sports_and_outdoors = sc.textFile('dbfs:/FileStore/amazon/metadata/meta_Sports_and_Outdoors.json')

# COMMAND ----------

cell_phones_and_accessories.getNumPartitions()
sports_and_outdoors.getNumPartitions()

# COMMAND ----------

cell_phones_and_accessories.coalesce(4).getNumPartitions()
sports_and_outdoors.coalesce(4).getNumPartitions()

# COMMAND ----------

cell_phones_and_accessories.coalesce(4).saveAsTextFile("dbfs:/FileStore/amazon/metadata/v2/meta_Cell_Phones_and_Accessories_v2")
sports_and_outdoors.coalesce(4).saveAsTextFile("dbfs:/FileStore/amazon/metadata/v2/meta_sports_and_outdoors")

# COMMAND ----------

meta_Cell_v2 = ps.read_json("/FileStore/amazon/metadata/v2/meta_Cell_Phones_and_Accessories_v2/*", lines=True)

sports_and_outdoors = ps.read_json("dbfs:/FileStore/amazon/metadata/v2/meta_sports_and_outdoors/*", lines=True)

# COMMAND ----------

meta_Cell_v2.info()  #590071개 행

# COMMAND ----------

display(meta_Cell_v2)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 변수 제거
# MAGIC - details, fit, imageURL, main_cat, tech1, tech2

# COMMAND ----------

meta_Cell_v2 = meta_Cell_v2.drop(columns=["details", "fit", "imageURL", "main_cat", "tech1", "tech2"])

sports_and_outdoors = sports_and_outdoors.drop(columns=["details", "fit", "imageURL", "main_cat", "tech1", "tech2"])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 결측값 null값으로 변경
# MAGIC - also_buy([]),  also_view([]),  brand(빈값, Unknown), description([]), feature([]),  imageURLHighRes([]), 
# MAGIC  similar_item(빈값),  title(빈값)
# MAGIC - 새로운 칼럼 생성안하고 기존 컬럼에 덮어씀

# COMMAND ----------

# []를 null로 바꾸는 함수
import pandas as pd
import numpy as np

def make_null_list(x):
    """
    []를 None을 반환하고, 그렇지 않으면 원래값
    """
    return None if x.size == 0 else x


# 함수 적용
meta_Cell_v2['also_buy'] = meta_Cell_v2['also_buy'].apply(make_null_list)
meta_Cell_v2['also_view'] = meta_Cell_v2['also_view'].apply(make_null_list)
meta_Cell_v2['description'] = meta_Cell_v2['description'].apply(make_null_list)
meta_Cell_v2['feature'] = meta_Cell_v2['feature'].apply(make_null_list)
meta_Cell_v2['imageURLHighRes'] = meta_Cell_v2['imageURLHighRes'].apply(make_null_list)

sports_and_outdoors['also_buy'] = sports_and_outdoors['also_buy'].apply(make_null_list)
sports_and_outdoors['also_view'] = sports_and_outdoors['also_view'].apply(make_null_list)
sports_and_outdoors['description'] = sports_and_outdoors['description'].apply(make_null_list)
sports_and_outdoors['feature'] = sports_and_outdoors['feature'].apply(make_null_list)
sports_and_outdoors['imageURLHighRes'] = sports_and_outdoors['imageURLHighRes'].apply(make_null_list)

# COMMAND ----------

# 빈값과 Unknown을 null로 바꾸는 함수
def make_null_brand(x):
    """
    문자열이 빈 문자열("") 또는 "Unknown"인 경우 None을 반환하고, 그렇지 않으면 원래의 문자열을 반환
    """
    return None if x == "" or x == "Unknown" else x


# 함수
meta_Cell_v2['brand'] = meta_Cell_v2['brand'].apply(make_null_brand)

sports_and_outdoors['brand'] = sports_and_outdoors['brand'].apply(make_null_brand)

# COMMAND ----------

# 빈 값을 null로 바꾸는 함수

def make_null_blank(x):
    """
    비어있으면 None을 반환하고, 그렇지 않으면 원래값
    """
    return None if x is None or len(x) == 0 else x

# 함수
meta_Cell_v2['similar_item'] = meta_Cell_v2['similar_item'].apply(make_null_blank)
meta_Cell_v2['title'] = meta_Cell_v2['title'].apply(make_null_blank)

sports_and_outdoors['similar_item'] = sports_and_outdoors['similar_item'].apply(make_null_blank)
sports_and_outdoors['title'] = sports_and_outdoors['title'].apply(make_null_blank)

# COMMAND ----------

# MAGIC %md
# MAGIC ### category 변수
# MAGIC - 대/중/소/세카테고리 새로운 칼럼 생성
# MAGIC - cat1, cat2, cat3, cat4
# MAGIC - 빈 값은 null로 생성

# COMMAND ----------

def extract_category(category, index):
    """
    category에서 지정된 인덱스에 해당하는 카테고리를 추출하고, 인덱스가 범위를 벗어나면 None을 반환
    """
    return category[index] if index < len(category) else None

# 함수 적용 예시
meta_Cell_v2['cat1'] = meta_Cell_v2['category'].apply(lambda x: extract_category(x, 0))
meta_Cell_v2['cat2'] = meta_Cell_v2['category'].apply(lambda x: extract_category(x, 1))
meta_Cell_v2['cat3'] = meta_Cell_v2['category'].apply(lambda x: extract_category(x, 2))
meta_Cell_v2['cat4'] = meta_Cell_v2['category'].apply(lambda x: extract_category(x, 3))

sports_and_outdoors['cat1'] = sports_and_outdoors['category'].apply(lambda x: extract_category(x, 0))
sports_and_outdoors['cat2'] = sports_and_outdoors['category'].apply(lambda x: extract_category(x, 1))
sports_and_outdoors['cat3'] = sports_and_outdoors['category'].apply(lambda x: extract_category(x, 2))
sports_and_outdoors['cat4'] = sports_and_outdoors['category'].apply(lambda x: extract_category(x, 3))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### asin 중복값 삭제

# COMMAND ----------

def remove_duplicate_rows(dataframe):
    columns_to_drop_duplicate = ['asin']
    dataframe = dataframe.drop_duplicates(subset=columns_to_drop_duplicate, keep='first')
    return dataframe

meta_Cell_v2 = remove_duplicate_rows(meta_Cell_v2)
sports_and_outdoors = remove_duplicate_rows(sports_and_outdoors)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### date 변수
# MAGIC - 날짜형식으로 변경(%Y-%m-%d)
# MAGIC - 원본 컬럼에 덮어쓰기
# MAGIC - 연,월 칼럼은 새로 생성 안함

# COMMAND ----------

meta_Cell_v2['date'] = ps.to_datetime(meta_Cell_v2['date'], errors='coerce')

sports_and_outdoors['date'] = ps.to_datetime(sports_and_outdoors['date'], errors='coerce')

# COMMAND ----------

meta_Cell_v2['date'] = meta_Cell_v2['date'].dt.strftime('%Y-%m-%d')

sports_and_outdoors['date'] = sports_and_outdoors['date'].dt.strftime('%Y-%m-%d')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### price 변수
# MAGIC - $3.87 -> 3.87
# MAGIC - $3.87 - $8.95 -> 평균
# MAGIC - 이상한 것&결측값 -> null
# MAGIC - new_price 열 생성

# COMMAND ----------

import re
import numpy as np

# price열에서 가격 추출
def extract_price(price_str):
    # '{'가 포함된 경우 null값
    if '{' in price_str:
        return np.nan
    # '$'와 '-'가 모두 포함된 경우
    elif '-' in price_str and '$' in price_str:
        # 숫자만 추출하여 평균값 계산
        prices = re.findall(r'\d+\.\d+', price_str)
        prices = [float(price) for price in prices]
        return round(np.mean(prices), 2) 
    # '$'만 포함된 경우
    elif '$' in price_str:
        # 숫자만 추출하여 반환
        return round(float(re.search(r'\d+\.\d+', price_str).group()), 2)
    # 그 외의 경우는 null값
    else:
        return np.nan


meta_Cell_v2['new_price'] = meta_Cell_v2['price'].apply(extract_price)

sports_and_outdoors['new_price'] = sports_and_outdoors['price'].apply(extract_price)

# COMMAND ----------



# COMMAND ----------

meta_Cell_v2.info()

# COMMAND ----------

sports_and_outdoors.info()

# COMMAND ----------

display(meta_Cell_v2.head(100))

# COMMAND ----------

display(sports_and_outdoors.head(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### rank 변수
# MAGIC - 우선 첫번째 숫자만 먼저 뽑아봄 (new_rank1)
# MAGIC - [{key1:value1,key2:value2}] 뽑는 건 추후

# COMMAND ----------

import re

def extract_first_number(text):
    # 쉼표를 제거한 후 숫자 추출
    match = re.search(r'\d+', text.replace(",", ""))
    if match:
        return int(match.group())
    else:
        return np.nan

meta_Cell_v2["new_rank1"] = meta_Cell_v2["rank"].apply(extract_first_number)


# COMMAND ----------

display(meta_Cell_v2)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### parquet 파일로 저장
# MAGIC - zstd 형식으로

# COMMAND ----------

meta_Cell_v2_fin=meta_Cell_v2
sports_and_outdoors_fin=sports_and_outdoors

# COMMAND ----------

path1 = "dbfs:/FileStore/amazon/metadata/v3/meta_Cell_Phones_and_Accessories_v3/"
path2 = "dbfs:/FileStore/amazon/metadata/v3/sports_and_outdoors_v3/"

# COMMAND ----------

meta_Cell_v2_fin.to_parquet('%s/to_parquet/meta_Cell_v3.parquet' % path1, compression='zstd')
sports_and_outdoors_fin.to_parquet('%s/to_parquet/sports_and_outdoors_v3.parquet' % path2, compression='zstd')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### parquet 불러오기

# COMMAND ----------

from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("example") \
    .getOrCreate()

# Parquet 파일 불러오기
path = "dbfs:/FileStore/amazon/metadata/v3/meta_Cell_Phones_and_Accessories_v3/to_parquet/meta_Cell_v3.parquet"
df = spark.read.parquet(path)

display(df)

# COMMAND ----------

path2 = "dbfs:/FileStore/amazon/metadata/v3/sports_and_outdoors_v3/to_parquet/sports_and_outdoors_v3.parquet"
df2 = spark.read.parquet(path)

# COMMAND ----------

display(df2)

# COMMAND ----------

# Write the data to a table.
table_name = "asac.meta_Cell_Phones_and_Accessories_fin"
df.write.saveAsTable(table_name)

# COMMAND ----------

table_name1 = "asac.sports_and_outdoors_fin"
df2.write.saveAsTable(table_name1)

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from asac.meta_Cell_Phones_and_Accessories_v2
