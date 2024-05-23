# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

# 테이블 읽기
cell_df = spark.read.table("asac.meta_cell_phones_and_accessories_new_price2")
sport_df = spark.read.table("asac.sports_and_outdoors_fin_v2")

# pyspark pandas DataFrame으로 변경
cell_df = ps.DataFrame(cell_df)
sport_df = ps.DataFrame(sport_df)

# COMMAND ----------

display(cell_df)

# COMMAND ----------

display(sport_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## cell_phone_and_accessories

# COMMAND ----------

len(cell_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC   min(new_price) AS min_value,
# MAGIC   max(new_price) AS max_value,
# MAGIC   avg(new_price) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.meta_cell_phones_and_accessories_new_price2;

# COMMAND ----------

plt.hist(cell_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()

# COMMAND ----------

plt.hist(cell_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price(log scale)')
plt.show()

# COMMAND ----------

plt.hist(cell_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.xscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price(x log scale)')
plt.show()

# COMMAND ----------

new_price_scaled = (cell_df['new_price'] - cell_df['new_price'].mean()) / cell_df['new_price'].std()

# 표준화된 값으로 히스토그램 그리기
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized New Price')
plt.show()

# COMMAND ----------

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Q3와 max_value 사이에 차이가 큼을 확인 -> 실제로 그런 것인지, 이상치인지 확인
# MAGIC -> asin 코드 이용해서 살펴보기
# MAGIC
# MAGIC - 작은 값들도 문제 없는지 확인하기
# MAGIC -> asin 코드 이용해서 살펴보기
# MAGIC
# MAGIC - 중간에 튄 값들도 확인하기 (400~600사이)

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,new_price, title from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price == 999.99  -- 95개/589356 -> 0.016% 

# COMMAND ----------

cell_df[cell_df['new_price']==999.99]["date"].value_counts(dropna=False)

# COMMAND ----------

sorted_counts = cell_df["new_price"].value_counts().sort_index(ascending=False)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC 999, 933.72, 924, 916, 900 확인

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,new_price, title from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price  between 900 and 999

# COMMAND ----------

# MAGIC %md
# MAGIC - $999.99 탐색 -> 검색 안되는 것들 존재
# MAGIC : B00PC1FYW6 : glass film(사용불가), B00WLJB8OK: 케이스(사용불가), B00WLJBAFM: 케이스(사용불가), B00WLJB8KYL: 케이스(사용불가), B01E70K6KW: 케이스(사용불가),
# MAGIC  B00RH0E4TE: armband(사용불가) -> 이상치로 판단(비슷한 상품들 가격 비교했을때) -> null로 수정한 new_price2열 생성
# MAGIC
# MAGIC - 999(4개 검색x, B004CU4UO6(사용불가)), 933.72(B00EJ4JDXS, $1032), 924(B01D5EQI30, 사용불가), 916(B018TGGH4E 검색x), 900(B01H4DOD9E(사용불가) 
# MAGIC -> 933.72는 제대로 검색이 되기 때문에, 999만 이상치로 판단 -> null로 수정한 new_price2열 생성

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE asac.meta_cell_phones_and_accessories_new_price2
# MAGIC SET new_price2 = NULL
# MAGIC WHERE new_price2 >= 999;

# COMMAND ----------

sorted_counts = cell_df["new_price2"].value_counts().sort_index(ascending=False)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC - Q1보다 작은 값 확인

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price<5.99

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, new_price2, title from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price2<5.99

# COMMAND ----------

sorted_counts = cell_df["new_price2"].value_counts().sort_index(ascending=True)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC - 가격 낮은 것 탐색
# MAGIC : B0004OPNTA(검색X), B000QJMW3S(검색X), B0197Q7FG4(사용불가), B019WUSADW(사용불가), B01BMWL7EW(검색X), B00Y1HAZG2(검색X), B012Y3FMQM(사용불가->$0.01 비슷한 물건 $6.95), B005EUQB4U(사용불가 ->$0.01 비슷한 물건 3.71$)
# MAGIC - 이상치를 IQR(하위만)로 생각해서 NULL값으로 변경하려고 했지만, 음수가 나옴
# MAGIC - z점수 3기준으로 하면, 낮은 값들이 포함이 안됨

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price2) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price2) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price2) AS q3,
# MAGIC   min(new_price2) AS min_value,
# MAGIC   max(new_price2) AS max_value,
# MAGIC   avg(new_price2) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.meta_cell_phones_and_accessories_new_price2;

# COMMAND ----------

down_num = 5.99 - 1.5*(12.99-5.99)
down_num # 음수 나옴

# COMMAND ----------

import pandas as pd
import numpy as np

cell_df['z_score'] = (cell_df['new_price2'] - cell_df['new_price2'].mean()) / cell_df['new_price2'].std()

# Z 점수가 3 이상인 행
outliers = cell_df[np.abs(cell_df['z_score']) >= 3]


# COMMAND ----------

col = ["new_price","z_score"]
display(cell_df[cell_df["z_score"]>=3][col])

# COMMAND ----------

col = ["new_price", "z_score"]
display(cell_df[cell_df["z_score"] >= 6][col])

# COMMAND ----------

col = ["new_price", "z_score"]
display(cell_df[(cell_df["z_score"] >= 6) & (cell_df["new_price"] <= 5)][col])

# COMMAND ----------

# MAGIC %md
# MAGIC - 임의로 $0.5 미만인 것들을 이상치로 취급 
# MAGIC -> 이를 null로 넣은 new_price2 열

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE asac.meta_cell_phones_and_accessories_new_price2
# MAGIC SET new_price2 = NULL
# MAGIC WHERE new_price2 < 0.5;

# COMMAND ----------

sorted_counts = cell_df["new_price2"].value_counts().sort_index(ascending=True)
display(sorted_counts)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price2) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price2) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price2) AS q3,
# MAGIC   min(new_price2) AS min_value,
# MAGIC   max(new_price2) AS max_value,
# MAGIC   avg(new_price2) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.meta_cell_phones_and_accessories_new_price2;

# COMMAND ----------

# MAGIC %md
# MAGIC - 중간에 튀는값 확인(400~600)

# COMMAND ----------

# 바꾸기 전 그래프
new_price_scaled = (cell_df['new_price'] - cell_df['new_price'].mean()) / cell_df['new_price'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

# 작은값, 큰 값 null으로 바꾼 후에 그래프
new_price_scaled2 = (cell_df['new_price2'] - cell_df['new_price2'].mean()) / cell_df['new_price2'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled2, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price2')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

plt.hist(cell_df["new_price2"], bins=50, color='blue', alpha=0.7)
plt.xlabel('New Price2')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('log y')
plt.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price2, asin, title from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price2 between 400 and 600

# COMMAND ----------

display(cell_df[(cell_df["new_price2"]>=400) & (cell_df["new_price2"]<=600)]["new_price2"].value_counts())

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price2, asin, title from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price2 == 500

# COMMAND ----------

# MAGIC %md
# MAGIC  - B00NU36XKU(유심, 사용불가), B00SBX9FSG(검색x), B00SLP4KBG(Headphone Adapter, 사용불가, title과 다름), B014KQ94A6(핸드폰, 사용불가), B01FJHDTEW(폰 화면 케이스, 사용불가), B01G5XHVHK(검색불가)
# MAGIC - 데이터는 이상하지만, 중간에 있는 값이라서 이상치를 취급하기도, 값을 바꾸기도 애매해서 그대로 사용

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 대카 기준 - 히스토그램, 박스플랏

# COMMAND ----------

plt.hist(cell_df["new_price2"], bins=50, color='blue', alpha=0.7)
plt.xlabel('New Price2')
plt.ylabel('Frequency')
plt.title('price2')
plt.show()

# COMMAND ----------

plt.hist(cell_df["new_price2"], bins=50, color='blue', alpha=0.7)
plt.xlabel('New Price2')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('log y')
plt.show()

# COMMAND ----------

new_price_scaled2 = (cell_df['new_price2'] - cell_df['new_price2'].mean()) / cell_df['new_price2'].std()
plt.hist(new_price_scaled2, bins=50, color='blue', alpha=0.7)
plt.xlabel('New Price2')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('log y, std x')
plt.show()

# COMMAND ----------

display(cell_df['new_price2'])

# COMMAND ----------

plt.figure(figsize=(8, 6)) 
plt.boxplot(cell_df['new_price2'].dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2') 
plt.show()

# COMMAND ----------

new_price_scaled2 = (cell_df['new_price2'] - cell_df['new_price2'].mean()) / cell_df['new_price2'].std()
plt.figure(figsize=(8, 6)) 
plt.boxplot(new_price_scaled2.dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2') 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 이상치 처리
# MAGIC - q1-1.5iqr
# MAGIC - q3+1.5iqr

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price2) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price2) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price2) AS q3,
# MAGIC   min(new_price2) AS min_value,
# MAGIC   max(new_price2) AS max_value,
# MAGIC   avg(new_price2) AS mean,
# MAGIC   stddev(new_price2) AS stddev
# MAGIC FROM asac.meta_cell_phones_and_accessories_new_price2;

# COMMAND ----------

iqr = 12.99-5.99
lower_bound = 5.99-1.5*iqr
upper_bound = 12.99+1.5*iqr
display(lower_bound,upper_bound) # 0보다 작기 때문에 upper_bound만 적용

# COMMAND ----------

no_out = cell_df[cell_df['new_price2']<23.5]['new_price2']
plt.figure(figsize=(8, 6)) 
plt.boxplot(no_out.dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2 no out') 
plt.show()

# COMMAND ----------

no_out = cell_df[cell_df['new_price2']<23.5]['new_price2']
new_price_scaled2 = (no_out - no_out.mean()) /no_out.std()
plt.figure(figsize=(8, 6)) 
plt.boxplot(new_price_scaled2.dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2 no out scale') 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 아웃라이어 제거, 스케일링 해도 얻을 정보는 없는 느낌

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 중카테고리

# COMMAND ----------

cell_df['cat2'].unique()

# COMMAND ----------

cell_df['cat2'].value_counts(dropna=False)

# COMMAND ----------

cat2_list = [cell_df['cat2'].unique()]

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat2,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price2) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price2) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price2) AS q3,
# MAGIC min(new_price2) AS min_value,
# MAGIC max(new_price2) AS max_value,
# MAGIC avg(new_price2) AS mean,
# MAGIC stddev(new_price2) AS stddev
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC GROUP BY cat2;

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price2, cat2, cat3, cat4, title, asin
# MAGIC from asac.meta_cell_phones_and_accessories_new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cat2,
# MAGIC     CASE 
# MAGIC         WHEN cat2 IS NULL THEN 'unknown'
# MAGIC         ELSE cat2 
# MAGIC     END AS cat22,
# MAGIC     new_price2 
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2;
# MAGIC
# MAGIC -- 순서가 바뀜
# MAGIC -- average : 셀폰 > 모바일 > 언논 > 심 > 케이스 > 악세사리
# MAGIC -- median :  셀폰 > 모바일 > 언논 > 케이스 > 악세사리 > 심
# MAGIC -- 심카드가 median으로 했을 때 순위가 낮아짐
# MAGIC -- Featured Categories는 가격이 없음

# COMMAND ----------

# MAGIC %md
# MAGIC 박스플랏을 태블로

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price2, cat2, cat3, cat4, title, asin
# MAGIC from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where cat2=="SIM Cards & Prepaid Minutes" and new_price2 > 700

# COMMAND ----------



# COMMAND ----------

cell_df["cat2"] = cell_df["cat2"].fillna("unknown")

# COMMAND ----------

cat2_list = cell_df["cat2"].unique().tolist()
cat2_list.pop()

# COMMAND ----------

cat2_list

# COMMAND ----------

fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[cell_df["cat2"] == i]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'{i}')

plt.show()

# COMMAND ----------

fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[cell_df["cat2"] == i]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')

plt.show()
plt.tight_layout()

# COMMAND ----------

# y축 통일
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[cell_df["cat2"] == i]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 0.15)  # y축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 이전의 아웃라이어 빼고 그래프 다시 확인해보기

# COMMAND ----------

iqr = 12.99-5.99
lower_bound = 5.99-1.5*iqr
upper_bound = 12.99+1.5*iqr
display(lower_bound,upper_bound) # 0보다 작기 때문에 upper_bound만 적용

# COMMAND ----------



# COMMAND ----------

# y축 통일, x축 통일
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[(cell_df["cat2"] == i) & (cell_df["new_price2"]<24)]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 1)  # y축 범위 설정
    axes[row, col].set_xlim(0, 25)  # x축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------

# y축 통일, x축 통일
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[(cell_df["cat2"] == i) & (cell_df["new_price2"]<24)]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=50, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 1)  # y축 범위 설정
    axes[row, col].set_xlim(0, 25)  # x축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------

# y축 통일, x축 통일
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[(cell_df["cat2"] == i) & (cell_df["new_price2"]<24)]['new_price2'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=10, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 1)  # y축 범위 설정
    axes[row, col].set_xlim(0, 25)  # x축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 추가 박스플랏은 태블로로

# COMMAND ----------

# MAGIC %md
# MAGIC ### 소카테고리

# COMMAND ----------

cell_df["cat3"].value_counts()

# COMMAND ----------

len(cell_df["cat3"].value_counts())

# COMMAND ----------

cat3_value_counts = cell_df["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts <= 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # 100개 이하인 소카가 516개

# COMMAND ----------

len(filtered_cat3_value_counts.index)

# COMMAND ----------

cat3_value_counts = cell_df["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts > 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # 25개도 많지만 최소한 나머지는 other 같이 묶어야 할 듯

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat3, new_price from asac.meta_Cell_Phones_and_Accessories_new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat3,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price2) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price2) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price2) AS q3,
# MAGIC min(new_price2) AS min_value,
# MAGIC max(new_price2) AS max_value,
# MAGIC avg(new_price2) AS mean,
# MAGIC stddev(new_price2) AS stddev
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC GROUP BY cat3;

# COMMAND ----------

# MAGIC %md
# MAGIC - 트리맵 생성
# MAGIC - 태블로에서 cat3 value_counts가 <=100인 것을 other로 만든 후, 박스플랏 생성

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## sports_and_outdoor

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC   min(new_price) AS min_value,
# MAGIC   max(new_price) AS max_value,
# MAGIC   avg(new_price) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2;

# COMMAND ----------

plt.hist(sport_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()

# COMMAND ----------

plt.hist(sport_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price(log scale)')
plt.show()

# COMMAND ----------

plt.hist(sport_df['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.xscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price(x log scale)')
plt.show()

# COMMAND ----------

new_price_scaled = (sport_df['new_price'] - sport_df['new_price'].mean()) / sport_df['new_price'].std()

# 표준화된 값으로 히스토그램 그리기
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.title('Histogram of Standardized New Price')
plt.show()

# COMMAND ----------

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Q3와 max_value 사이에 차이가 큼을 확인 -> 실제로 그런 것인지, 이상치인지 확인
# MAGIC -> asin 코드 이용해서 살펴보기
# MAGIC
# MAGIC - 작은 값들도 문제 없는지 확인하기
# MAGIC -> asin 코드 이용해서 살펴보기
# MAGIC
# MAGIC - cell과 달리 중간에 튀는 값들 존재x

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,new_price, title from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price == 999.99  -- 94개/957217 -> 0.009% 

# COMMAND ----------

sport_df[sport_df['new_price']==999.99]["date"].value_counts(dropna=False)

# COMMAND ----------

sorted_counts = sport_df["new_price"].value_counts().sort_index(ascending=False)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC 999~999.99 확인
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,new_price, title from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price >= 999

# COMMAND ----------

# MAGIC %sql
# MAGIC -- new_price2 열 생성 및 값 설정
# MAGIC ALTER TABLE asac.sports_and_outdoors_fin_v2
# MAGIC ADD COLUMN new_price2 INT;
# MAGIC
# MAGIC -- new_price2 열에 값 설정
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price2 = CASE
# MAGIC                     WHEN new_price >= 999 THEN NULL
# MAGIC                     ELSE new_price
# MAGIC                 END;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- new_price3 열 생성 및 값 설정
# MAGIC ALTER TABLE asac.sports_and_outdoors_fin_v2
# MAGIC ADD COLUMN new_price3 float;
# MAGIC
# MAGIC -- new_price3 열에 값 설정
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price3 = CASE
# MAGIC                     WHEN new_price >= 999 THEN NULL
# MAGIC                     ELSE new_price
# MAGIC                 END;

# COMMAND ----------

# MAGIC %sql
# MAGIC select max(new_price3) from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

sport_df = spark.read.table("asac.sports_and_outdoors_fin_v2")
sport_df = ps.DataFrame(sport_df)

# COMMAND ----------

sorted_counts = sport_df["new_price3"].value_counts().sort_index(ascending=False)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC Q1보다 작은 값 확인(11.67)

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price<11.67

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, new_price2, title from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price3<11.67

# COMMAND ----------

sorted_counts = sport_df["new_price3"].value_counts().sort_index(ascending=True)
display(sorted_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC - 이상치를 IQR(하위만)로 생각해서 NULL값으로 변경하려고 했지만, 음수가 나옴

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price3) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price3) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price3) AS q3,
# MAGIC   min(new_price3) AS min_value,
# MAGIC   max(new_price3) AS max_value,
# MAGIC   avg(new_price3) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2

# COMMAND ----------

down_num = 11.66 - 1.5*(45.84-11.66)
down_num # 음수 나옴

# COMMAND ----------

# MAGIC %md
# MAGIC - 현재 스포츠 아웃도어에서 가장 싼 가격 3.88
# MAGIC -> 임의로 3미만인 것을 이상치로 판단

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price3 = NULL
# MAGIC WHERE new_price3 < 3;

# COMMAND ----------

sorted_counts = sport_df["new_price3"].value_counts().sort_index(ascending=True)
display(sorted_counts)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price3) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price3) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price3) AS q3,
# MAGIC   min(new_price3) AS min_value,
# MAGIC   max(new_price3) AS max_value,
# MAGIC   avg(new_price3) AS mean,
# MAGIC   stddev(new_price) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2;

# COMMAND ----------

sport_df = spark.read.table("asac.sports_and_outdoors_fin_v2")
sport_df = ps.DataFrame(sport_df)

# COMMAND ----------

# MAGIC %md
# MAGIC - 대카기준 - 히스토그램, 박스플랏

# COMMAND ----------

# 바꾸기 전 그래프
new_price_scaled = (sport_df['new_price'] - sport_df['new_price'].mean()) / sport_df['new_price'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

display(sport_df['new_price3'].min())

# COMMAND ----------

# 작은값, 큰 값 null으로 바꾼 후에 그래프
new_price_scaled2 = (sport_df['new_price3'] - sport_df['new_price3'].mean()) / sport_df['new_price3'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled2, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price3')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

plt.hist(sport_df["new_price3"], bins=50, color='blue', alpha=0.7)
plt.xlabel('New Price3')
plt.ylabel('Frequency')
plt.title('price3')
plt.show()

# COMMAND ----------

plt.figure(figsize=(8, 6)) 
plt.boxplot(sport_df['new_price3'].dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price3') 
plt.show()

# COMMAND ----------

new_price_scaled2 = (sport_df['new_price3'] - sport_df['new_price3'].mean()) / sport_df['new_price3'].std()
plt.figure(figsize=(8, 6)) 
plt.boxplot(new_price_scaled2.dropna())
plt.title('Box Plot of new_price3')
plt.ylabel('New Price2') 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 이상치처리

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price3) AS q1,
# MAGIC   percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price3) AS median,
# MAGIC   percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price3) AS q3,
# MAGIC   min(new_price3) AS min_value,
# MAGIC   max(new_price3) AS max_value,
# MAGIC   avg(new_price3) AS mean,
# MAGIC   stddev(new_price3) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2;

# COMMAND ----------

iqr = 46.57 - 11.99
lower_bound = 11.99-1.5*iqr
upper_bound = 46.57+1.5*iqr
display(lower_bound,upper_bound) # 0보다 작기 때문에 upper_bound만 적용

# COMMAND ----------

no_out = sport_df[sport_df['new_price3']<98.44]['new_price3']
plt.figure(figsize=(8, 6)) 
plt.boxplot(no_out.dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price3 no out') 
plt.show()

# COMMAND ----------



# COMMAND ----------

no_out = sport_df[sport_df['new_price3']<98.44]['new_price3']
new_price_scaled2 = (no_out - no_out.mean()) /no_out.std()
plt.figure(figsize=(8, 6)) 
plt.boxplot(new_price_scaled2.dropna())
plt.title('Box Plot of new_price3')
plt.ylabel('New Price3 no out scale') 
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 중카테고리

# COMMAND ----------

sport_df['cat2'].unique()

# COMMAND ----------

sport_df['cat2'].value_counts(dropna=False)

# COMMAND ----------

cat2_list2 = [sport_df['cat2'].unique()]

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat2,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price3) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price3) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price3) AS q3,
# MAGIC min(new_price3) AS min_value,
# MAGIC max(new_price3) AS max_value,
# MAGIC avg(new_price3) AS mean,
# MAGIC stddev(new_price3) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2
# MAGIC GROUP BY cat2;

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price3, cat2, cat3, cat4, title, asin
# MAGIC from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cat1, cat2,
# MAGIC     CASE 
# MAGIC         WHEN cat2 IS NULL THEN 'unknown'
# MAGIC         ELSE cat2 
# MAGIC     END AS cat22,
# MAGIC     new_price3 
# MAGIC FROM asac.sports_and_outdoors_fin_v2;
# MAGIC
# MAGIC -- 순서가 바뀜
# MAGIC -- average : 스포츠 피트니스 > 아웃도어 레크레이션 > 언논 > 팬샵
# MAGIC -- median :  스포츠 피트니스 >  팬샵> 아웃도어 레크레이션 > 언논 
# MAGIC -- 팬샵의 median값이 커짐

# COMMAND ----------

# MAGIC %md
# MAGIC 박스플랏은 태블로로

# COMMAND ----------

sport_df["cat2"] = sport_df["cat2"].fillna("unknown")

# COMMAND ----------

cat2_list = sport_df["cat2"].unique().tolist()

# COMMAND ----------

cat2_list

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = sport_df[sport_df["cat2"] == i]['new_price3'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'{i}')

plt.show()

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = sport_df[sport_df["cat2"] == i]['new_price3'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')

plt.show()
plt.tight_layout()

# COMMAND ----------

# y축 통일
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = sport_df[sport_df["cat2"] == i]['new_price3'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 0.15)  # y축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 아웃라이어 빼고 그래프 다시 확인

# COMMAND ----------

# y축 통일, x축 통일
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = sport_df[(sport_df["cat2"] == i) & (sport_df["new_price3"]<98.44)]['new_price3'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black', density=True)
    axes[row, col].set_title(f'{i}')
    axes[row, col].set_ylim(0, 1)  # y축 범위 설정
    axes[row, col].set_xlim(0, 25)  # x축 범위 설정

plt.tight_layout()
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 소카테고리

# COMMAND ----------

sport_df["cat3"].value_counts()

# COMMAND ----------

len(sport_df["cat3"].value_counts())

# COMMAND ----------

cat3_value_counts = sport_df["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts <= 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # 100개 이하인 소카가 130개

# COMMAND ----------

len(filtered_cat3_value_counts.index)

# COMMAND ----------

cat3_value_counts = sport_df["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts > 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # 37개도 많지만 최소한 나머지는 other 같이 묶어야 할 듯

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat1, cat2, cat3, asin, new_price3 from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat3,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price3) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price3) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price3) AS q3,
# MAGIC min(new_price3) AS min_value,
# MAGIC max(new_price3) AS max_value,
# MAGIC avg(new_price3) AS mean,
# MAGIC stddev(new_price3) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2
# MAGIC GROUP BY cat3;

# COMMAND ----------

# MAGIC %md
# MAGIC - 트리맵 생성
# MAGIC - 태블로에서 cat3 value_counts가 <=100인 것을 other로 만든 후, 박스플랏 생성

# COMMAND ----------



# COMMAND ----------


