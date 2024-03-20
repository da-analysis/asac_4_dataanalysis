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

meta_cell_sc = sc.textFile("dbfs:/FileStore/amazon/metadata/meta_Cell_Phones_and_Accessories.json")
import json
ar_json = meta_cell_sc.map(json.loads)
from pyspark.sql.types import StringType, StructType, StructField, FloatType, LongType, MapType, BooleanType

feature_schema = StructType([
    StructField("also_buy", StringType()),
    StructField("also_view", StringType()),
    StructField("asin", StringType()),
    StructField("brand", StringType()),
    StructField("category", StringType()),
    StructField("date", StringType()),
    StructField("description", StringType()),
    StructField("details", StringType()),
    StructField("feature", StringType()),
    StructField("fit", StringType()),
    StructField("imageURL", StringType()),
    StructField("imageURLHighRes", StringType()),
    StructField("main_cat", StringType()),
    StructField("price", StringType()),
    StructField("rank", StringType()),
    StructField("similar_item", StringType()),
    StructField("tech1", StringType()),
    StructField("tech2", StringType()),
    StructField("title", StringType()),
])

meta_cell_df = spark.createDataFrame(ar_json, feature_schema) 
meta_cell_df.createOrReplaceTempView("meta_cell_df")
display(meta_cell_df)

# COMMAND ----------

meta_cell_df.printSchema()

# COMMAND ----------

meta_cell_df = meta_cell_df.select(
    col('also_buy'),col('also_view'),col('asin'), col('brand'), col('category'),col('date'),
    col('description'),col('details') ,col('feature'),col("fit"),col("imageURL"),col("imageURLHighRes"),col("main_cat"),col("main_cat"),col("price"),col("rank"),col("similar_item"),col("tech1"),col("tech2"),col("title")
)

# COMMAND ----------

display(meta_cell_df)

# COMMAND ----------

display(meta_cell_df.head())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC pyspark.pandas 활용

# COMMAND ----------

file_path ="dbfs:/FileStore/amazon/metadata/meta_Cell_Phones_and_Accessories.json"

# COMMAND ----------

meta_cell = ps.read_json(file_path,lines=True)

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

display(meta_cell.head(100))

# COMMAND ----------

meta_cell.head()

# COMMAND ----------

display(meta_cell.tail())

# COMMAND ----------

# MAGIC %md
# MAGIC - 저번에 한 것처럼 + 중,소 카테고리 개수, 가격, 결측치 개수, 브랜드 유니크 개수
# MAGIC - 날짜 남기기
# MAGIC - 랭킹 처리 방법 고민하기 (대카/중카...)
# MAGIC - 가격 피쳐 (range 어떻게 할지?)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1. also_buy

# COMMAND ----------

meta_cell["also_buy"].describe()

# COMMAND ----------

meta_cell["also_buy"].head(10)

# COMMAND ----------

meta_cell["also_buy"].tail(10)

# COMMAND ----------

meta_cell["also_buy"].value_counts().head(5)
# []로 들어간 것이 545808개

# COMMAND ----------

545808/590071
# 92.5%

# COMMAND ----------

# also_buy 결측값([])이 92.5% 이지만, 다른식으로 활용할 수 있으니 그대로 가져가기
# null값 쉽게 확인하기 위해서 new_also_buy에서 []인 값을 null로 생성

# COMMAND ----------

first_also_buy = meta_cell.loc[0, "also_buy"]

if first_also_buy.size == 0:
    print("True")
else:
    print("False")

# COMMAND ----------

meta_cell['new_also_buy'] = meta_cell['also_buy'].apply(lambda x: None if x.size == 0 else x)
# pyspark.pandas에서는 nan이나 null을 직접 안넣음 -> None값 넣음

meta_cell.head()

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

display(meta_cell["new_also_buy"].head())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2. also_view

# COMMAND ----------

meta_cell["also_view"].describe()

# COMMAND ----------

meta_cell["also_view"].head(10)

# COMMAND ----------

meta_cell["also_view"].tail(10)

# COMMAND ----------

meta_cell["also_view"].value_counts().head(5)
# []로 들어간 것이 541574개

# COMMAND ----------

541574/590071

# COMMAND ----------

# also_view 결측값([])이 91.8% 이지만, 다른식으로 활용할 수 있으니 그대로 가져가기
# null값 쉽게 확인하기 위해서 new_also_view에서 []인 값을 null로 생성

# COMMAND ----------

meta_cell['new_also_view'] = meta_cell['also_view'].apply(lambda x: None if x.size == 0 else x)

display(meta_cell.head(10))

# COMMAND ----------

display(meta_cell.info())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3. asin

# COMMAND ----------

meta_cell["asin"].describe()

# COMMAND ----------

meta_cell["asin"].head(10)

# COMMAND ----------

meta_cell["asin"].tail(10)

# COMMAND ----------

meta_cell["asin"].value_counts(1)

# COMMAND ----------

# asin이 2글자 이하인 행
short_asin_rows = meta_cell[meta_cell['asin'].str.len() <= 2]
short_asin_rows
# 없음 -> 다 들어갔다고 생각

# COMMAND ----------

# asin이 10글자 dksls 행
ten_asin_rows = meta_cell[meta_cell['asin'].str.len() != 10]
ten_asin_rows
# 없음 -> 다 들어갔다고 생각

# COMMAND ----------

meta_cell["asin"].value_counts().head(20)

# COMMAND ----------

value_counts = meta_cell["asin"].value_counts()
count_above_2 = (value_counts >= 2).sum()
count_above_2

# COMMAND ----------

temp = meta_cell[meta_cell["asin"] == "B00004Y87Z"]

temp[["asin", "also_view", "brand", "date","category","description"]]


# COMMAND ----------

temp = meta_cell[meta_cell["asin"] == "B00004Y87Z"]

temp[["asin", "also_view", "brand", "date","category","description"]]

# COMMAND ----------

filtered_rows = meta_cell[meta_cell["asin"] == "B00004Y87Z"]
filtered_rows

# COMMAND ----------

filtered_rows = meta_cell[(meta_cell["asin"] == "B00004Y87Z") | (meta_cell["asin"] == "B00005LLYE")]

filtered_rows

# COMMAND ----------

asin_counts = meta_cell["asin"].value_counts()

temp = asin_counts[asin_counts >= 2].index.tolist()

# COMMAND ----------

temp

# COMMAND ----------

len(temp)

# COMMAND ----------

len(set(temp))

# COMMAND ----------

filtered_rows = meta_cell[meta_cell["asin"].isin(temp)]

display(filtered_rows)

# COMMAND ----------

len(filtered_rows)

# COMMAND ----------

temp_1 = ["B00004Y87Z","B0000505VE","B0000505TJ"]
filtered_rows_1 = meta_cell[meta_cell["asin"].isin(temp_1)]

display(filtered_rows_1)

# COMMAND ----------

temp_1 = ["B00004Y87Z"]
filtered_rows_1 = meta_cell[meta_cell["asin"].isin(temp_1)]

display(filtered_rows_1)

# COMMAND ----------

# 그대로 가져가기 (결측치x, 2글자 이하인 행 x, 10글자 아닌 행 x), 2번 나타난 값들 존재0.1%(715/590071) -> 다른 열들의 값들도 동일한 것으로 보임 -> 추후 카테고리 선택 후 asin중복되는 값들 확인 후, 다른 칼럼 값들이 동일해보인다면, 중복되는 행들중 하나만 선택
#-> 지금을 일단 다 가져감

# COMMAND ----------

# MAGIC %md
# MAGIC 4. brand
# MAGIC - 유니크 갯수

# COMMAND ----------

meta_cell["brand"].describe()

# COMMAND ----------

meta_cell["brand"].head(10)

# COMMAND ----------

meta_cell["brand"].tail(10)

# COMMAND ----------

meta_cell["brand"].value_counts(5)

# COMMAND ----------

meta_cell["brand"].value_counts(3)

# COMMAND ----------

meta_cell["brand"].value_counts()

# COMMAND ----------

(12711+5040)/590071

# COMMAND ----------

len(meta_cell["brand"].value_counts())

# COMMAND ----------

len(meta_cell["brand"].value_counts(dropna=False))

# COMMAND ----------

len(meta_cell["brand"].value_counts(dropna=True))

# COMMAND ----------

meta_cell['new_brand'] = meta_cell['brand'].apply(lambda x: None if x=="" or x == "Unknown" else x)

# 결과 출력
display(meta_cell.head(10))

# COMMAND ----------

meta_cell["new_brand"].value_counts()

# COMMAND ----------

len(meta_cell["new_brand"].value_counts(dropna=False))

# COMMAND ----------

len(meta_cell["new_brand"].value_counts(dropna=True))

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

# 결측값과 Unknown 데이터 합쳐서 3%(12711+5040), 혹시 Unknown이라는 브랜드가 있나 확인했지만 없는듯함, 유니크한 브랜드 개수(43215(43217-2)개), 빈값과 Unknown값을 결측으로 인식하는 새로운 칼럼 new_brand 생성 97%(572320/590071)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5. category

# COMMAND ----------

meta_cell["category"].describe()

# COMMAND ----------

meta_cell["category"].head(10)

# COMMAND ----------

meta_cell["category"].tail(10)

# COMMAND ----------

meta_cell["category"].value_counts(1)

# COMMAND ----------

meta_cell['cat1'] = meta_cell['category'].apply(lambda x: x[0] if len(x) >= 1 else None)
meta_cell['cat2'] = meta_cell['category'].apply(lambda x: x[1] if len(x) >= 2 else None)
meta_cell['cat3'] = meta_cell['category'].apply(lambda x: x[2] if len(x) >= 3 else None)
meta_cell['cat4'] = meta_cell['category'].apply(lambda x: x[3] if len(x) >= 4 else None)

display(meta_cell.head(10))

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

print(590071-534466,(590071-534466)/590071)
print(590071-534466,(590071-534466)/590071)
print(590071-337796,(590071-337796)/590071)
print(590071-40869,(590071-40869)/590071)

# COMMAND ----------

# 대/중/소/세 칼럼(4개) 나누기, 원본 칼럼 두기, 각 칼럼별 결측값(9%(55605)/ 9%(55605)/ 43%(252275)/ 93%(549202)) -> 비율 기준은 단순 전체 수

# COMMAND ----------

# MAGIC %md
# MAGIC 6. date
# MAGIC - 연, 월 컬럼 나눈 후에 원래 날짜 컬럼도 남겨두기

# COMMAND ----------

org = ps.read_json(file_path,lines=True)

# COMMAND ----------

# ops_on_diff_frames 옵션을 활성화하여 다른 데이터프레임에서의 연산을 허용
ps.options.compute.ops_on_diff_frames = True

# meta_cell 데이터프레임의 "date" 열을 org 데이터프레임의 "date" 열 값으로 대체
meta_cell['date'] = org['date']

# ops_on_diff_frames 옵션을 다시 비활성화하여 연산 허용을 되돌림
ps.options.compute.ops_on_diff_frames = False

# COMMAND ----------

meta_cell["date"].describe()

# COMMAND ----------

meta_cell["date"].describe()

# COMMAND ----------

meta_cell["date"].head(10)

# COMMAND ----------

meta_cell["date"].tail(10)

# COMMAND ----------

meta_cell["date"].value_counts().head(3)

# COMMAND ----------

meta_cell["date"].value_counts().head(3)

# COMMAND ----------

meta_cell['new_date'] = ps.to_datetime(meta_cell['date'], errors='coerce')

# COMMAND ----------

display(meta_cell.head(10))

# COMMAND ----------

display(meta_cell.head(20))

# COMMAND ----------

meta_cell["new_date"].value_counts().head(3)

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

# 'year' 열 생성
meta_cell['year'] = meta_cell['new_date'].dt.year.fillna('').astype(int)


# COMMAND ----------

# 'month' 열 생성
meta_cell['month'] = meta_cell['new_date'].dt.month.fillna('').astype(int)


display(meta_cell.head(20))

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

# 결측값 92.2%(544191/590071), 날짜 형식이 아닌 것들 0.6%(3378) 존재, 날짜형식으로 바꾼 new_date 컬럼 생성 7.2%(42502/590071), 연(year),월(month) 컬럼 새로 생성

# COMMAND ----------

# MAGIC %md
# MAGIC 7. description

# COMMAND ----------

meta_cell["description"].describe()

# COMMAND ----------

meta_cell["description"].head(10)

# COMMAND ----------

meta_cell["description"].tail(10)

# COMMAND ----------

meta_cell["description"].value_counts().head(3)

# COMMAND ----------

meta_cell['new_description'] = meta_cell['description'].apply(lambda x: None if x.size == 0 else x)

display(meta_cell.head(10))

# COMMAND ----------

# []로 들어가 있는 결측값 32.3%(190852/590071), []를 null값으로 변경한 new_description 컬럼 생성, /n,<br> 같은 것들이 있음 -> 추후 활용할 경우 얘기해보며 삭제 고려

# COMMAND ----------

# MAGIC %md
# MAGIC 8. details 

# COMMAND ----------

meta_cell["details"].describe()

# COMMAND ----------

meta_cell["details"].head(10)

# COMMAND ----------

meta_cell["details"].tail(10)

# COMMAND ----------

meta_cell["details"].value_counts().head(3)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["details"])

# COMMAND ----------

# 같은 데이터가 98%(578167/590071) -> 제거하기

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 9. feature 

# COMMAND ----------

meta_cell["feature"].value_counts().head(2)

# COMMAND ----------

198873/590071

# COMMAND ----------

meta_cell['new_feature'] = meta_cell['feature'].apply(lambda x: None if x.size == 0 else x)

display(meta_cell.head(10))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 10. fit 

# COMMAND ----------

meta_cell["fit"].value_counts().head(2)

# COMMAND ----------

590028/590071

# COMMAND ----------

# 결측값이 99.9% 이므로 제거

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["fit"])

# COMMAND ----------

meta_cell.info()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 11. imageURL

# COMMAND ----------

meta_cell["imageURL"].describe()

# COMMAND ----------

meta_cell["imageURL"].head(10)

# COMMAND ----------

meta_cell["imageURL"].tail(10)

# COMMAND ----------

meta_cell["imageURL"].value_counts().head(3)

# COMMAND ----------

meta_cell = meta_cell.drop(columns = ["imageURL"])

# COMMAND ----------

# MAGIC %md
# MAGIC 12. imageURLHighRes

# COMMAND ----------

meta_cell["imageURLHighRes"].describe()

# COMMAND ----------

meta_cell["imageURLHighRes"].head(10)

# COMMAND ----------

meta_cell["imageURLHighRes"].tail(10)

# COMMAND ----------

meta_cell["imageURLHighRes"].value_counts().head(3)

# COMMAND ----------

meta_cell['new_image'] = meta_cell['imageURLHighRes'].apply(lambda x: None if x.size == 0 else x)
display(meta_cell.head(10))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 13. main_cat

# COMMAND ----------

meta_cell["main_cat"].describe()

# COMMAND ----------

meta_cell["main_cat"].head(10)

# COMMAND ----------

meta_cell["main_cat"].tail(10)

# COMMAND ----------

meta_cell["main_cat"].value_counts().head(3)

# COMMAND ----------

meta_cell["main_cat"].value_counts().head(10)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["main_cat"])

# COMMAND ----------

# 여전히 잘못들어간 카테고리들 존재 & category 컬럼에서 충분히 정보 가지고 있음 -> 제거

# COMMAND ----------

# MAGIC %md
# MAGIC 14. price

# COMMAND ----------

meta_cell["price"].describe()

# COMMAND ----------

meta_cell["price"].head(10)

# COMMAND ----------

meta_cell["price"].tail(10)

# COMMAND ----------

meta_cell["price"].value_counts().head(3)

# COMMAND ----------

meta_cell["price"].value_counts().head(10)

# COMMAND ----------

filtered_rows = meta_cell[meta_cell['price'].str.contains(r'\$.*-', regex=True, na=False)]


row_value = filtered_rows['price'].iloc[0]
row_value


# COMMAND ----------

filtered_rows['price'].value_counts().head(5)

# COMMAND ----------

len(filtered_rows['price'])

# COMMAND ----------

import re
import numpy as np

# price열에서 가격 추출하는 함수 정의
def extract_price(price_str):
    # '{'가 포함된 경우 null값으로 처리
    if '{' in price_str:
        return np.nan
    # '$'와 '-'가 모두 포함된 경우
    elif '-' in price_str and '$' in price_str:
        # 숫자만 추출하여 평균값 계산
        prices = re.findall(r'\d+\.\d+', price_str)
        prices = [float(price) for price in prices]
        return round(np.mean(prices), 2)  # 소수점 둘째 자리까지 반올림하여 반환
    # '$'만 포함된 경우
    elif '$' in price_str:
        # 숫자만 추출하여 반환
        return round(float(re.search(r'\d+\.\d+', price_str).group()), 2)  # 소수점 둘째 자리까지 반올림하여 반환
    # 그 외의 경우는 null값으로 처리
    else:
        return np.nan

# new_price열에 함수 적용하여 생성
meta_cell['new_price'] = meta_cell['price'].apply(extract_price)

# COMMAND ----------

col =["price","new_price"]
display(meta_cell[col].head(500))

# COMMAND ----------

col =["price","new_price"]
display(meta_cell[col].tail(500))

# COMMAND ----------

filtered_rows = meta_cell[meta_cell['price'].str.contains('\$.*-', regex=True, na=False)]

filtered_data = filtered_rows[['price', 'new_price']]
display(filtered_data.head(100))


# COMMAND ----------

meta_cell["new_price"].value_counts().head(3)

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

# 결측값 78.4%(462507/590071), 아예 이상한 데이터 존재, 범위 데이터 존재(126/590071)
# $3.87 -> 3.87로 변경하고, $3.87 - $8.95 -> 평균으로 변경, 이상한 것&결측값 -> null값으로 변경한 new_price열 생성
# new_price 결측값 아닌 것 -> 21.1%(124721/590071)

# COMMAND ----------

# MAGIC %md
# MAGIC 15. rank

# COMMAND ----------

meta_cell["rank"].describe()

# COMMAND ----------

meta_cell["rank"].head(10)

# COMMAND ----------

meta_cell["rank"].tail(10)

# COMMAND ----------

meta_cell["rank"].value_counts().head(3)

# COMMAND ----------

display(meta_cell["rank"].value_counts().head(20))

# COMMAND ----------

meta_cell["rank"].value_counts().tail(20)

# COMMAND ----------



# COMMAND ----------

ps.options.compute.ops_on_diff_frames = True

meta_cell['rank'] = org['rank']

ps.options.compute.ops_on_diff_frames = False

# COMMAND ----------

meta_cell['new_rank'] = meta_cell['rank'].str.replace('\s{2,}', '', regex=True)  # 공백이 2칸 이상 있는 것 삭제

# COMMAND ----------

display(meta_cell[["rank","new_rank"]].head(100))

# COMMAND ----------

import pandas as pd

meta_cell['new_rank'] = meta_cell['rank'].str.extract('(\d+)').astype(int)


# COMMAND ----------

display(meta_cell[["rank","new_rank"]].head(100))

# COMMAND ----------

meta_cell['new_rank'] = meta_cell['rank'].apply(lambda x: None if x.size == 0 else x)

meta_cell.head()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 16. similar_item

# COMMAND ----------

meta_cell["similar_item"].describe()

# COMMAND ----------

meta_cell["similar_item"].head(10)

# COMMAND ----------

meta_cell["similar_item"].tail(10)

# COMMAND ----------

meta_cell["similar_item"].value_counts().head(2)

# COMMAND ----------

display(meta_cell.head(100))

# COMMAND ----------

display(meta_cell.tail(100))

# COMMAND ----------

meta_cell['new_similar_item'] = meta_cell['similar_item'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell['new_similar_item'] = meta_cell['similar_item'].apply(lambda x: None if x is None or len(x) == 0 else x)


# COMMAND ----------

display(meta_cell.head(10))

# COMMAND ----------

meta_cell.info()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 17. tech1

# COMMAND ----------

meta_cell["tech1"].value_counts(5)

# COMMAND ----------

meta_cell["tech1"].value_counts().head(2)

# COMMAND ----------

583486/590071

# COMMAND ----------

# 결측값이 98.9% 이므로 제거

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["tech1"])

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

# MAGIC %md
# MAGIC 18. tech2

# COMMAND ----------

meta_cell["tech2"].value_counts(5)

# COMMAND ----------

meta_cell["tech2"].value_counts().head(2)

# COMMAND ----------

589840/590071

# COMMAND ----------

# 결측값이 99.9% 이므로 제거

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["tech2"])

# COMMAND ----------

meta_cell.info()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 19. title 

# COMMAND ----------

meta_cell["title"].describe()

# COMMAND ----------

meta_cell["title"].head(10)

# COMMAND ----------

meta_cell["title"].tail(10)

# COMMAND ----------

meta_cell["title"].value_counts().head(3)

# COMMAND ----------

display(meta_cell.head(100))

# COMMAND ----------

count_empty_title = meta_cell[meta_cell['title'].str.len() == 0]

display(count_empty_title)

# COMMAND ----------

meta_cell['new_title'] = meta_cell['title'].apply(lambda x: None if x is None or len(x) == 0 else x)
display(meta_cell.head(100))


# COMMAND ----------

null_title_rows = meta_cell[meta_cell['new_title'].isnull()]
display(null_title_rows)

# COMMAND ----------

meta_cell.info()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 최종 전처리 파일 생성

# COMMAND ----------

import pandas as pd
import json
import numpy as np
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import pyspark.pandas as ps

# COMMAND ----------

file_path ="dbfs:/FileStore/amazon/metadata/meta_Cell_Phones_and_Accessories.json"

# COMMAND ----------

meta_cell = ps.read_json(file_path,lines=True)

# COMMAND ----------

meta_cell['new_also_buy'] = meta_cell['also_buy'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell['new_also_view'] = meta_cell['also_view'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell['new_brand'] = meta_cell['brand'].apply(lambda x: None if x=="" or x == "Unknown" else x)

# COMMAND ----------

meta_cell['cat1'] = meta_cell['category'].apply(lambda x: x[0] if len(x) >= 1 else None)
meta_cell['cat2'] = meta_cell['category'].apply(lambda x: x[1] if len(x) >= 2 else None)
meta_cell['cat3'] = meta_cell['category'].apply(lambda x: x[2] if len(x) >= 3 else None)
meta_cell['cat4'] = meta_cell['category'].apply(lambda x: x[3] if len(x) >= 4 else None)


# COMMAND ----------

meta_cell['new_date'] = ps.to_datetime(meta_cell['date'], errors='coerce')


# COMMAND ----------

meta_cell['year'] = meta_cell['new_date'].dt.year.fillna('').astype(int)


# COMMAND ----------

meta_cell['month'] = meta_cell['new_date'].dt.month.fillna('').astype(int)


# COMMAND ----------

meta_cell['new_description'] = meta_cell['description'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["details"])


# COMMAND ----------

meta_cell['new_feature'] = meta_cell['feature'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["fit"])


# COMMAND ----------

meta_cell = meta_cell.drop(columns = ["imageURL"])


# COMMAND ----------

meta_cell['new_image'] = meta_cell['imageURLHighRes'].apply(lambda x: None if x.size == 0 else x)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["main_cat"])


# COMMAND ----------

import re
import numpy as np

# price열에서 가격 추출하는 함수 정의
def extract_price(price_str):
    # '{'가 포함된 경우 null값으로 처리
    if '{' in price_str:
        return np.nan
    # '$'와 '-'가 모두 포함된 경우
    elif '-' in price_str and '$' in price_str:
        # 숫자만 추출하여 평균값 계산
        prices = re.findall(r'\d+\.\d+', price_str)
        prices = [float(price) for price in prices]
        return round(np.mean(prices), 2)  # 소수점 둘째 자리까지 반올림하여 반환
    # '$'만 포함된 경우
    elif '$' in price_str:
        # 숫자만 추출하여 반환
        return round(float(re.search(r'\d+\.\d+', price_str).group()), 2)  # 소수점 둘째 자리까지 반올림하여 반환
    # 그 외의 경우는 null값으로 처리
    else:
        return np.nan
# new_price열에 함수 적용하여 생성
meta_cell['new_price'] = meta_cell['price'].apply(extract_price)

# COMMAND ----------

meta_cell['new_rank'] = meta_cell['rank'].str.replace('\s{2,}', '', regex=True)  # 공백이 2칸 이상 있는 것 삭제

# COMMAND ----------

meta_cell['new_similar_item'] = meta_cell['similar_item'].apply(lambda x: None if x is None or len(x) == 0 else x)

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["tech1"])

meta_cell = meta_cell.drop(columns=["tech2"])

# COMMAND ----------

meta_cell['new_title'] = meta_cell['title'].apply(lambda x: None if x is None or len(x) == 0 else x)

# COMMAND ----------

meta_cell.info()

# COMMAND ----------

display(meta_cell.head(100)) # new_rank 아직 처리안한 최종 데이터셋

# COMMAND ----------

meta_cell["rank2"] = meta_cell["rank"].replace(",", "")

# COMMAND ----------

display(meta_cell["rank2"].head(10))

# COMMAND ----------

import re

def extract_first_number(text):
    # 쉼표를 제거한 후 숫자 추출
    match = re.search(r'\d+', text.replace(",", ""))
    if match:
        return int(match.group())
    else:
        return np.nan

# "rank" 열에서 첫 번째 숫자를 추출하여 "new_rank" 열 생성
meta_cell["new_rank"] = meta_cell["rank"].apply(extract_first_number)

# COMMAND ----------

display(meta_cell.head(100))

# COMMAND ----------

meta_cell = meta_cell.drop(columns=["rank2"])

# COMMAND ----------

display(meta_cell.head(100))

# COMMAND ----------

meta_cell.info()

# COMMAND ----------


