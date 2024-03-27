# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.pandas as ps

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Delta Lake to PySpark Pandas DataFrame") \
    .getOrCreate()

# Delta Lake 테이블 읽기
delta_table_path = "dbfs:/user/hive/warehouse/asac.db/meta_cell_phones_and_accessories_fin"
df = spark.read.format("delta").load(delta_table_path)

# Spark DataFrame을 PySpark Pandas DataFrame으로 변환
pdf_cell = ps.DataFrame(df)

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Delta Lake to PySpark Pandas DataFrame") \
    .getOrCreate()

# Delta Lake 테이블 읽기
delta_table_path = "dbfs:/user/hive/warehouse/asac.db/sports_and_outdoors_fin_v2"
df = spark.read.format("delta").load(delta_table_path)

# Spark DataFrame을 PySpark Pandas DataFrame으로 변환
pdf_sport = ps.DataFrame(df)

# COMMAND ----------

display(pdf_cell)

# COMMAND ----------

display(pdf_sport)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.meta_cell_phones_and_accessories_fin

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.sports_and_outdoors_fin

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - cell_phones_and_accessories

# COMMAND ----------

len(pdf_cell)

# COMMAND ----------

# MAGIC %sql
# MAGIC select new_price from asac.meta_cell_phones_and_accessories_fin

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
# MAGIC FROM asac.meta_cell_phones_and_accessories_fin;

# COMMAND ----------

# MAGIC %md
# MAGIC Q3와 max_value 사이에 차이가 큼을 확인 -> 실제로 그런 것인지, 이상치인지

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.meta_cell_phones_and_accessories_fin
# MAGIC where new_price > 500   # 146개

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from asac.meta_cell_phones_and_accessories_fin
# MAGIC where new_price == 999.99  # 95개/589356 -> 0.016% 

# COMMAND ----------

# MAGIC %md
# MAGIC 999.99값인 애들 title 변수에 있는 것을 검색해보니까, 대부분 핸드폰 케이스, 러닝할 때, 팔에 끼우는 것 -> 아무리 비싸도 20$ 이내
# MAGIC -> 9.99가 잘못 들어 간 것이 아닌가
# MAGIC
# MAGIC 이 외에 895.9, 686.7, 674.4 -> 차량 어댑터,헤드더  -> 비싼 것들임 -> 이것은 제대로 들어간 것으로 생각
# MAGIC
# MAGIC -> 그럼 999.99를 -> 9.99로 바꿀 것인지 -> 일단 new_price2로 999.99인 애들을 null로 바꾸고 new_price와 new_price2 분포 모두 확인해보기
# MAGIC -> 아예 새로 델타 파일을 생성했음 -> 그냥 sql로 만들어도 될듯

# COMMAND ----------

pdf_cell['new_price2'] = pdf_cell['new_price'].where(pdf_cell['new_price'] != 999.99, None)

# COMMAND ----------

display(pdf_cell[pdf_cell['new_price'] == 999.99])

# COMMAND ----------

path1 = "dbfs:/FileStore/amazon/metadata/v4/meta_Cell_Phones_and_Accessories_v4/"

pdf_cell.to_parquet('%s/to_parquet/meta_Cell_v4.parquet' % path1, compression='zstd')

# COMMAND ----------

from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("example") \
    .getOrCreate()

# Parquet 파일 불러오기
path = "dbfs:/FileStore/amazon/metadata/v4/meta_Cell_Phones_and_Accessories_v4/to_parquet/meta_Cell_v4.parquet"
df = spark.read.parquet(path)

# COMMAND ----------

table_name = "asac.meta_Cell_Phones_and_Accessories_new_price2"
df.write.saveAsTable(table_name)

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
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2;

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Delta Lake to PySpark Pandas DataFrame") \
    .getOrCreate()

# Delta Lake 테이블 읽기
delta_table_path = "dbfs:/user/hive/warehouse/asac.db/meta_cell_phones_and_accessories_new_price2"
df = spark.read.format("delta").load(delta_table_path)

# Spark DataFrame을 PySpark Pandas DataFrame으로 변환
pdf_cell_2 = ps.DataFrame(df)

# COMMAND ----------

display(pdf_cell_2[pdf_cell_2['new_price2'] == 999])

# COMMAND ----------

sorted_counts = pdf_cell_2["new_price2"].value_counts().sort_index(ascending=False)

# COMMAND ----------

display(sorted_counts)  # 가격 높은 순으로 정렬한 value_counts

# COMMAND ----------

# MAGIC %md
# MAGIC 여전히 큰 값들 존재 -> 우선 그럼 new_price2말고 new_price로 탐색 진행 -> 회의후에 결정

# COMMAND ----------

display(pdf_cell)

# COMMAND ----------

import matplotlib.pyplot as plt

plt.hist(pdf_cell_2['new_price'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()  # 아주 많이 치우쳐서 존재함

# COMMAND ----------

plt.hist(pdf_cell_2['new_price'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()  # 빈 크기 늘렸을 때

# COMMAND ----------

plt.hist(pdf_cell_2['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()  # 빈 크기 늘렸을 때

# COMMAND ----------

# 로그 스케일 
plt.hist(pdf_cell_2['new_price'], bins=10, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()

# 900~1000구간이 갑자기 튀는 느낌이 있긴함 -> 999.99처리안한 데이터씀 -> 그럼 null로 하는게 나은지?(new_price2)

# COMMAND ----------

# 로그 스케일 , bins = 100
plt.hist(pdf_cell_2['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()   # 위의 bins = 10보다 확확 줄어드는 느낌이 적어짐


# COMMAND ----------

# 999.99 -> null값으로 변경한 것 (new_price2로 한 것) -> 확실히 1000 근처 구간 튀는것 줄어듦
# 로그 스케일 , bins = 100 
import matplotlib.pyplot as plt
plt.hist(pdf_cell_2['new_price2'], bins=100, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price2 (Log Scale)')
plt.show()

# COMMAND ----------

# density

plt.hist(pdf_cell_2['new_price'], bins=10, color='skyblue', edgecolor='black', density=True)
plt.xlabel('New Price')
plt.ylabel('Density')
plt.title('Density Plot of New Price')
plt.show()

# COMMAND ----------

plt.hist(pdf_cell_2['new_price'], bins=50, color='skyblue', edgecolor='black', density=True)
plt.xlabel('New Price')
plt.ylabel('Density')
plt.title('Density Plot of New Price')
plt.show()

# COMMAND ----------

plt.hist(pdf_cell_2['new_price'], bins=100, color='skyblue', edgecolor='black', density=True)
plt.yscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 중카 기준

# COMMAND ----------

pdf_cell_2['cat2'].unique()

# COMMAND ----------

pdf_cell_2['cat2'].value_counts(dropna=False)

# COMMAND ----------

print(pdf_cell_2.columns)

# COMMAND ----------

display(pdf_cell_2)

# COMMAND ----------

pdf_cell_2['cat2'].unique()

# COMMAND ----------

num = len(pdf_cell_2['cat2'].unique())

# COMMAND ----------

cat2_list = [pdf_cell_2['cat2'].unique()]

# COMMAND ----------

cat2_list[0][0]

# COMMAND ----------

pdf_cell_2['cat2'].unique()

# COMMAND ----------

pdf_cell_2[pdf_cell_2["cat2"]=="Cell Phones"]["new_price"].value_counts()

# COMMAND ----------

display(pdf_cell_2[pdf_cell_2["cat2"]=="Cell Phones"]["new_price"])

# COMMAND ----------

cat2_list[0][0]

# COMMAND ----------

for i in range(num):
    display(cat2_list[0][i])

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat2,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC min(new_price) AS min_value,
# MAGIC max(new_price) AS max_value,
# MAGIC avg(new_price) AS mean,
# MAGIC stddev(new_price) AS stddev
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC GROUP BY cat2;

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat2, new_price from asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC -- 평균 -> cat2가 null인 애들의 평균도 알아야 할 까 -> 그럼 null -> unknown으로 나타낸 cat22컬럼 만들기

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cat2,
# MAGIC     CASE 
# MAGIC         WHEN cat2 IS NULL THEN 'unknown'
# MAGIC         ELSE cat2 
# MAGIC     END AS cat22,
# MAGIC     new_price 
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2;
# MAGIC
# MAGIC -- 순서가 바뀜
# MAGIC -- average : 셀폰 > 모바일 > 언논 > 심 > 케이스 > 악세사리
# MAGIC -- median :  셀폰 > 모바일 > 언논 > 케이스 > 악세사리 > 심
# MAGIC -- 심카드가 median으로 했을 때 순위가 낮아짐

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cat2,
# MAGIC     CASE 
# MAGIC         WHEN cat2 IS NULL THEN 'unknown'
# MAGIC         ELSE cat2 
# MAGIC     END AS cat22,
# MAGIC     new_price 
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC GROUP BY cat2, CASE WHEN cat2 IS NULL THEN 'unknown' ELSE cat2 END, new_price;

# COMMAND ----------



# COMMAND ----------

pdf_cell_2['cat2'].unique()

# COMMAND ----------

pdf_cell_2["cat2"] = pdf_cell_2["cat2"].fillna("unknown")

# COMMAND ----------

pdf_cell_2['cat2'].value_counts()

# COMMAND ----------

cat2_list = pdf_cell_2["cat2"].unique().tolist()
cat2_list

# COMMAND ----------

for i in cat2_list:
    print(i)

# COMMAND ----------

fig, axes = plt.subplots(1, 7, figsize=(30,10))
j = 0
for i in cat2_list:
    filtered_data = pdf_cell_2[pdf_cell_2["cat2"]==i]['new_price'].dropna()
    axes[j].hist(filtered_data, bins=100, color='skyblue', edgecolor='black')
    axes[j].set_title(f'{i}')
    j += 1
plt.show()

# COMMAND ----------

fig, axes = plt.subplots(4, 2, figsize=(20, 10))

for j, i in enumerate(cat2_list):
    filtered_data = pdf_cell_2[pdf_cell_2["cat2"] == i]['new_price'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].hist(filtered_data, bins=100, color='skyblue', edgecolor='black')
    axes[row, col].set_title(f'{i}')

plt.show()

# COMMAND ----------

fig, axes = plt.subplots(4, 2, figsize=(20, 10))

for j, i in enumerate(cat2_list):
    filtered_data = pdf_cell_2[pdf_cell_2["cat2"] == i]['new_price'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].boxplot(filtered_data)
    axes[row, col].set_title(f'{i}')

plt.show()
plt.tight_layout()

# COMMAND ----------

plt.figure(figsize=(8, 6))
pdf_cell_2.boxplot(column='new_price', by='cat2')
plt.title('Boxplot of Value by cat2')
plt.xlabel('cat2')
plt.ylabel('new_price')
plt.show()

# COMMAND ----------



# COMMAND ----------

plt.hist(pdf_cell_2[pdf_cell_2["cat2"]=="unknown"]['new_price'], bins=100, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('cat2 =unknown (Log Scale)')
plt.show()

# COMMAND ----------

plt.hist(pdf_cell_2['new_price'], bins=100, color='skyblue', edgecolor='black', density=True)
plt.yscale('log')
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 소카기준

# COMMAND ----------

pdf_cell_2["cat3"].value_counts()

# COMMAND ----------

len(pdf_cell_2["cat3"].value_counts())

# COMMAND ----------

cat3_value_counts = pdf_cell_2["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts <= 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts)

# COMMAND ----------

cat3_value_counts = pdf_cell_2["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts > 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # 25개도 많지만 최소한 나머지는 other 같이 묶어야 할 듯

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat3 from asac.meta_Cell_Phones_and_Accessories_new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat3, new_price from asac.meta_Cell_Phones_and_Accessories_new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat3,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC min(new_price) AS min_value,
# MAGIC max(new_price) AS max_value,
# MAGIC avg(new_price) AS mean,
# MAGIC stddev(new_price) AS stddev
# MAGIC FROM asac.meta_Cell_Phones_and_Accessories_new_price2
# MAGIC GROUP BY cat3;

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - sports_and_outdoor

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

# MAGIC %sql
# MAGIC select * from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price == 999.99  # 94개

# COMMAND ----------

sorted_counts = pdf_sport["new_price"].value_counts().sort_index(ascending=False)
display(sorted_counts)

# COMMAND ----------

display(pdf_sport[pdf_sport['new_price'] == 999.99])

# COMMAND ----------

import matplotlib.pyplot as plt

plt.hist(pdf_sport['new_price'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('New Price')
plt.ylabel('Frequency')
plt.title('Histogram of New Price')
plt.show()  # 아주 많이 치우쳐서 존재함

# COMMAND ----------

# 로그 스케일  -> cellphone보다 900~1000 구간이 많이 튀지는 않음 -> 그래도 증가는 했음
plt.hist(pdf_sport['new_price'], bins=10, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()

# COMMAND ----------

plt.hist(pdf_sport['new_price'], bins=50, color='skyblue', edgecolor='black')
plt.yscale('log') 
plt.xlabel('New Price')
plt.ylabel('Frequency (Log Scale)')
plt.title('Histogram of New Price (Log Scale)')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 중카

# COMMAND ----------

pdf_sport["cat2"].value_counts(dropna=False)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat2,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC min(new_price) AS min_value,
# MAGIC max(new_price) AS max_value,
# MAGIC avg(new_price) AS mean,
# MAGIC stddev(new_price) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2
# MAGIC GROUP BY cat2;

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat2, new_price from asac.sports_and_outdoors_fin_v2;  ---> null값도 특징 있을 수 있으니 unknown으로 바꾼 cat22열 만들기

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     cat2,
# MAGIC     CASE 
# MAGIC         WHEN cat2 IS NULL THEN 'unknown'
# MAGIC         ELSE cat2 
# MAGIC     END AS cat22,
# MAGIC     new_price 
# MAGIC FROM asac.sports_and_outdoors_fin_v2;   -- average와 median에 차이 발생 -> 피트니스>레크>언논>팬샵 / 피트니스>팬샵>레크>언논 --> 팬샵이 위로 올라옴

# COMMAND ----------

pdf_sport["cat2"] = pdf_sport["cat2"].fillna("unknown")

# COMMAND ----------

cat2_list_sp = pdf_sport["cat2"].unique().tolist()

# COMMAND ----------

cat2_list_sp

# COMMAND ----------

fig, axes = plt.subplots(1, 4, figsize=(30,10))
j = 0
for i in cat2_list_sp:
    filtered_data = pdf_sport[pdf_sport["cat2"]==i]['new_price'].dropna()
    axes[j].hist(filtered_data, bins=100, color='skyblue', edgecolor='black')
    axes[j].set_title(f'{i}')
    j += 1
plt.show()

# COMMAND ----------

fig, axes = plt.subplots(1, 4, figsize=(30,10))
j = 0
for i in cat2_list_sp:
    filtered_data = pdf_sport[pdf_sport["cat2"]==i]['new_price'].dropna()
    axes[j].boxplot(filtered_data)
    axes[j].set_title(f'{i}')
    j += 1
plt.show()

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(20, 20))

for j, i in enumerate(cat2_list_sp):
    filtered_data = pdf_sport[pdf_sport["cat2"] == i]['new_price'].dropna()
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].boxplot(filtered_data)
    axes[row, col].set_title(f'{i}')

plt.show()

# COMMAND ----------



# COMMAND ----------

plt.figure(figsize=(8, 6))
pdf_sport.boxplot(column='new_price', by='cat2')
plt.title('Boxplot of Value by cat2')
plt.xlabel('cat2')
plt.ylabel('new_price')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 소카

# COMMAND ----------

pdf_sport["cat3"].value_counts()

# COMMAND ----------

len(pdf_sport["cat3"].value_counts())

# COMMAND ----------

cat3_value_counts = pdf_sport["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts <= 100]
filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts)

# COMMAND ----------

cat3_value_counts = pdf_sport["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts > 100]
filtered_cat3_value_counts

# COMMAND ----------

filtered_cat3_value_counts

# COMMAND ----------

len(filtered_cat3_value_counts) # -> 100개 넘는 것도 37개나 됨 -> 더 줄여도 될듯함

# COMMAND ----------

cat3_value_counts = pdf_sport["cat3"].value_counts()
filtered_cat3_value_counts = cat3_value_counts[cat3_value_counts <= 100]
left_values = filtered_cat3_value_counts.index.tolist()
left_values

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat3 from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat3, new_price from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT cat3,
# MAGIC percentile_cont(0.25) WITHIN GROUP (ORDER BY new_price) AS q1,
# MAGIC percentile_cont(0.5) WITHIN GROUP (ORDER BY new_price) AS median,
# MAGIC percentile_cont(0.75) WITHIN GROUP (ORDER BY new_price) AS q3,
# MAGIC min(new_price) AS min_value,
# MAGIC max(new_price) AS max_value,
# MAGIC avg(new_price) AS mean,
# MAGIC stddev(new_price) AS stddev
# MAGIC FROM asac.sports_and_outdoors_fin_v2
# MAGIC GROUP BY cat3;

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


