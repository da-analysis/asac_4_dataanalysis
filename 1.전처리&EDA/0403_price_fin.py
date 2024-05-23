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
# MAGIC # price 이상치 처리

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

new_price_scaled = (cell_df['new_price'] - cell_df['new_price'].mean()) / cell_df['new_price'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 양극단의 percentile 지정해서 new_price2 값을 null로 지정
# MAGIC SELECT 
# MAGIC   percentile_cont(0.01) WITHIN GROUP (ORDER BY new_price) AS p1,
# MAGIC   percentile_cont(0.001) WITHIN GROUP (ORDER BY new_price) AS p01,
# MAGIC   percentile_cont(0.0001) WITHIN GROUP (ORDER BY new_price) AS p001,
# MAGIC   percentile_cont(0.99) WITHIN GROUP (ORDER BY new_price) AS p99,
# MAGIC   percentile_cont(0.999) WITHIN GROUP (ORDER BY new_price) AS p999,
# MAGIC   percentile_cont(0.9999) WITHIN GROUP (ORDER BY new_price) AS p9999
# MAGIC
# MAGIC FROM asac.meta_cell_phones_and_accessories_new_price2
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price<=0.99

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price>=89.95

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price<=0.01

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from asac.meta_cell_phones_and_accessories_new_price2
# MAGIC where new_price>=664.776

# COMMAND ----------

# 1,000분위수 활용해서 0.01이하, 664.776이상인 가격은 null값으로 처리하는 new_price2열 생성

# COMMAND ----------

# MAGIC %sql
# MAGIC -- new_price2 다시 조정
# MAGIC UPDATE asac.meta_cell_phones_and_accessories_new_price2
# MAGIC SET new_price2 = new_price

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE asac.meta_cell_phones_and_accessories_new_price2
# MAGIC SET new_price2 = CASE
# MAGIC                     WHEN new_price <= 0.01  THEN NULL
# MAGIC                     WHEN new_price >= 664.776  THEN NULL
# MAGIC                     ELSE new_price
# MAGIC                 END;

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

cell_df = spark.read.table("asac.meta_cell_phones_and_accessories_new_price2")
cell_df = ps.DataFrame(cell_df)

new_price_scaled = (cell_df['new_price2'] - cell_df['new_price2'].mean()) / cell_df['new_price2'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------

len(sport_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,cat2,cat3,new_price, title from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price == 999.99  -- 94개/957217 -> 0.009% 

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,title,new_price,cat1,cat2 from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price == 999.99 and brand == "The Elastos"

# COMMAND ----------

sorted_counts = sport_df["new_price"].value_counts().sort_index()
display(sorted_counts)

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin, brand,date,cat2,cat3,new_price, title from asac.sports_and_outdoors_fin_v2
# MAGIC where new_price == 0.00 or new_price == 0.01 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 양극단의 percentile 지정해서 new_price2 값을 null로 지정
# MAGIC SELECT 
# MAGIC   percentile_cont(0.01) WITHIN GROUP (ORDER BY new_price) AS p1,
# MAGIC   percentile_cont(0.001) WITHIN GROUP (ORDER BY new_price) AS p01,
# MAGIC   percentile_cont(0.0001) WITHIN GROUP (ORDER BY new_price) AS p001,
# MAGIC   percentile_cont(0.99) WITHIN GROUP (ORDER BY new_price) AS p99,
# MAGIC   percentile_cont(0.999) WITHIN GROUP (ORDER BY new_price) AS p999,
# MAGIC   percentile_cont(0.9999) WITHIN GROUP (ORDER BY new_price) AS p9999
# MAGIC
# MAGIC FROM asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# MAGIC %sql
# MAGIC -- new_price2 다시 조정
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price2 = new_price

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price2 = CASE
# MAGIC                     WHEN new_price <= 1.41  THEN NULL
# MAGIC                     WHEN new_price >= 881.668  THEN NULL
# MAGIC                     ELSE new_price
# MAGIC                 END;

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
# MAGIC FROM asac.sports_and_outdoors_fin_v2;

# COMMAND ----------

sport_df = spark.read.table("asac.sports_and_outdoors_fin_v2")
sport_df = ps.DataFrame(sport_df)

new_price_scaled = (sport_df['new_price2'] - sport_df['new_price2'].mean()) / sport_df['new_price2'].std()

# x표준화하고, y는 로그 스케일 같이 적용한 것
plt.hist(new_price_scaled, bins=50, color='blue', alpha=0.7)
plt.xlabel('Standardized New Price')
plt.ylabel('Frequency')
plt.yscale('log')
plt.title('std x, log y')
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 카테고리별 박스플랏 숫자표시 (대카)

# COMMAND ----------

# MAGIC %md
# MAGIC ## cellphone

# COMMAND ----------

plt.figure(figsize=(8, 6)) 
plt.boxplot(cell_df['new_price2'].dropna())
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2') 
plt.show()

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

import matplotlib.pyplot as plt

# 이상치 제외한 데이터 추출
no_out = cell_df[cell_df['new_price2'] < 23.5]['new_price2'].dropna()

plt.figure(figsize=(10, 10))
plt.boxplot(no_out, whis=[0, 100], showmeans=True, showfliers=False, showcaps=False, showbox=True)

stats = ['Q1', 'Q2', 'Q3', 'Min', 'Max',"Mean"]
positions = [1, 2, 3, 4, 5, 6]

for stat, pos in zip(stats, positions):
    value = None
    if stat == 'Q1':
        value = no_out.quantile(0.25)
    elif stat == 'Q2':
        value = no_out.median()
    elif stat == 'Q3':
        value = no_out.quantile(0.75)
    elif stat == 'Min':
        value = no_out.min()
    elif stat == 'Max':
        value = no_out.max()
    elif stat == 'Mean':
        value = round(no_out.mean(),2)
    plt.text(1, value, f'{stat}: {value}', fontsize=15, va='center', ha='center', color='blue')

# 그래프 제목 및 축 레이블 설정
plt.title('Box Plot of new_price2')
plt.ylabel('New Price2 no out')
plt.show()


# COMMAND ----------

no_out.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------

# MAGIC %sql
# MAGIC -- new_price2 다시 조정
# MAGIC UPDATE asac.sports_and_outdoors_fin_v2
# MAGIC SET new_price3 = new_price2

# COMMAND ----------

plt.figure(figsize=(8, 6)) 
plt.boxplot(sport_df['new_price3'].dropna())
plt.title('Box Plot of new_price3')
plt.ylabel('New Price3') 
plt.show()

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

iqr = 45.76-11.7
lower_bound = 11.7-1.5*iqr
upper_bound =  45.76+1.5*iqr
display(lower_bound)
display(upper_bound) # 0보다 작기 때문에 upper_bound만 적용

# COMMAND ----------

import matplotlib.pyplot as plt

# 이상치 제외한 데이터 추출
no_out = sport_df[sport_df['new_price3'] < 96.85]['new_price3'].dropna()

plt.figure(figsize=(10, 10))
plt.boxplot(no_out, whis=[0, 100], showmeans=True, showfliers=False, showcaps=False, showbox=True)

stats = ['Q1', 'Q2', 'Q3', 'Min', 'Max',"Mean"]
positions = [1, 2, 3, 4, 5, 6]

for stat, pos in zip(stats, positions):
    value = None
    if stat == 'Q1':
        value = no_out.quantile(0.25)
    elif stat == 'Q2':
        value = no_out.median()
    elif stat == 'Q3':
        value = no_out.quantile(0.75)
    elif stat == 'Min':
        value = no_out.min()
    elif stat == 'Max':
        value = no_out.max()
    elif stat == 'Mean':
        value = round(no_out.mean(),2)
    plt.text(1, value, f'{stat}: {round(value,2)}', fontsize=15, va='center', ha='center', color='blue')

# 그래프 제목 및 축 레이블 설정
plt.title('Box Plot of new_price2')
plt.ylabel('New Price3 no out')
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 중카 누적합계

# COMMAND ----------

# MAGIC %md
# MAGIC ## cellphone

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

cell_df["cat2"] = cell_df["cat2"].fillna("unknown")
cat2_list = cell_df["cat2"].unique().tolist()
cat2_list.pop() #'Featured Categories'는 가격 없음

fig, axes = plt.subplots(3, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = cell_df[cell_df["cat2"] == i]['new_price2'].dropna()
    sorted_data = np.sort(filtered_data.to_numpy())  # Series를 NumPy 배열로 변환
    cumulative = np.arange(len(sorted_data)) / len(sorted_data)
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].plot(sorted_data, cumulative, color='blue')
    axes[row, col].set_title(f'{i}')

plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

cell_df["cat2"] = cell_df["cat2"].fillna("unknown")
cat2_list = cell_df["cat2"].unique().tolist()
cat2_list.pop() #'Featured Categories'는 가격 없음

fig, ax = plt.subplots(figsize=(10, 6))

for i in cat2_list:
    filtered_data = cell_df[cell_df["cat2"] == i]['new_price2'].dropna()
    sorted_data = np.sort(filtered_data.to_numpy())  # Series를 NumPy 배열로 변환
    cumulative = np.arange(len(sorted_data)) / len(sorted_data)
    ax.plot(sorted_data, cumulative, label=i)

ax.set_title('Cumulative Distribution of new_price2')
ax.set_xlabel('New Price2')
ax.set_ylabel('Cumulative Probability')
ax.legend()
plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------



# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

sport_df["cat2"] = sport_df["cat2"].fillna("unknown")
cat2_list = sport_df["cat2"].unique().tolist()

fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for j, i in enumerate(cat2_list):
    filtered_data = sport_df[sport_df["cat2"] == i]['new_price3'].dropna()
    sorted_data = np.sort(filtered_data.to_numpy())  # Series를 NumPy 배열로 변환
    cumulative = np.arange(len(sorted_data)) / len(sorted_data)
    row = j // 2  # 행 인덱스
    col = j % 2   # 열 인덱스
    axes[row, col].plot(sorted_data, cumulative, color='blue')
    axes[row, col].set_title(f'{i}')

plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

sport_df["cat2"] = sport_df["cat2"].fillna("unknown")
cat2_list = sport_df["cat2"].unique().tolist()

fig, ax = plt.subplots(figsize=(10, 6))

for i in cat2_list:
    filtered_data = sport_df[sport_df["cat2"] == i]['new_price3'].dropna()
    sorted_data = np.sort(filtered_data.to_numpy())  # Series를 NumPy 배열로 변환
    cumulative = np.arange(len(sorted_data)) / len(sorted_data)
    ax.plot(sorted_data, cumulative, label=i)

ax.set_title('Cumulative Distribution of new_price3')
ax.set_xlabel('New Price3')
ax.set_ylabel('Cumulative Probability')
ax.legend()
plt.show()


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 소카 (특정 중카 선택해서)

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat2,cat3,cat4,new_price2 from asac.meta_cell_phones_and_accessories_new_price2

# COMMAND ----------

# MAGIC %sql
# MAGIC select cat2,cat3,cat4,new_price3 from asac.sports_and_outdoors_fin_v2

# COMMAND ----------

# 태블로 확인

# COMMAND ----------

sport_df[sport_df["cat2"]=="Sports & Fitness"]["cat3"].value_counts().head(5)

# COMMAND ----------

col = ["cat2","new_price3"]
display(sport_df[sport_df["cat3"]=="Hunting & Fishing"][col])

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE asac.sports_and_outdoors_fin_v2
# MAGIC DROP COLUMN new_price2;

# COMMAND ----------

display(sport_df[sport_df["cat3"]=="Hunting & Fishing"]['new_price3'])

# COMMAND ----------

sport_df[sport_df["cat2"]=="Fan Shop"]["cat3"].value_counts().head(5)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # also_buy, also_view 길이 차이/ 길이 분포

# COMMAND ----------

# MAGIC %md
# MAGIC ## cellphone

# COMMAND ----------

display(cell_df)

# COMMAND ----------

len(cell_df["also_buy"][56])

# COMMAND ----------

import matplotlib.pyplot as plt

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths = cell_df['also_buy'].dropna().apply(len)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of List Lengths in also_buy Column')
plt.xlabel('Length of List')
plt.ylabel('Frequency')
plt.show()


# COMMAND ----------

max(lengths.to_numpy())

# COMMAND ----------

import matplotlib.pyplot as plt

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths = cell_df['also_view'].dropna().apply(len)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of List Lengths in also_view Column')
plt.xlabel('Length of List')
plt.ylabel('Frequency')
plt.show()


# COMMAND ----------

cell_df.info()

# COMMAND ----------

display(cell_df['also_view'].isnull().sum(), " "  ,cell_df['also_buy'].isnull().sum())

# COMMAND ----------

# 통계표 작성해보기
import numpy as np
import pandas as pd

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths = cell_df['also_view'].dropna().apply(len)

# 길이에 대한 빈도수를 crosstab으로 계산
frequency_table = pd.crosstab(index=lengths.to_numpy(), columns='Frequency')

display(frequency_table)

# COMMAND ----------

import numpy as np
import pandas as pd

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths_also_view = cell_df['also_view'].dropna().apply(len)
lengths_also_buy = cell_df['also_buy'].dropna().apply(len)

# 길이에 대한 빈도수를 crosstab으로 계산
frequency_table_also_view = pd.crosstab(index=lengths_also_view.to_numpy(), columns='cell (also_view)')
frequency_table_also_buy = pd.crosstab(index=lengths_also_buy.to_numpy(), columns='cell (also_buy)')

# 빈도표 합치기
frequency_table = pd.concat([frequency_table_also_view, frequency_table_also_buy], axis=1)

display(frequency_table)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------

display(sport_df)

# COMMAND ----------

import matplotlib.pyplot as plt

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths = sport_df['also_buy'].dropna().apply(len)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of List Lengths in also_buy Column')
plt.xlabel('Length of List')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

max(lengths.to_numpy())

# COMMAND ----------

import matplotlib.pyplot as plt

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths = sport_df['also_view'].dropna().apply(len)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of List Lengths in also_view Column')
plt.xlabel('Length of List')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

max(lengths.to_numpy())

# COMMAND ----------

import numpy as np
import pandas as pd

# None 값이 아닌 행만 필터링하여 리스트 길이 계산
lengths_also_view = sport_df['also_view'].dropna().apply(len)
lengths_also_buy = sport_df['also_buy'].dropna().apply(len)

# 길이에 대한 빈도수를 crosstab으로 계산
frequency_table_also_view2 = pd.crosstab(index=lengths_also_view.to_numpy(), columns='sport (also_view)')
frequency_table_also_buy2 = pd.crosstab(index=lengths_also_buy.to_numpy(), columns='sport (also_buy)')

# 빈도표 합치기
frequency_table = pd.concat([frequency_table_also_view,frequency_table_also_buy,frequency_table_also_view2, frequency_table_also_buy2], axis=1)

display(frequency_table)

# COMMAND ----------

# 0인 애들은 제외한거
import matplotlib.pyplot as plt
lengths_also_view = cell_df['also_view'].dropna().apply(len)
lengths_also_buy = cell_df['also_buy'].dropna().apply(len)
lengths_also_view2 = sport_df['also_view'].dropna().apply(len)
lengths_also_buy2 = sport_df['also_buy'].dropna().apply(len)

# 각 열에 대한 데이터 추출
data_also_view = lengths_also_view.to_numpy()
data_also_buy = lengths_also_buy.to_numpy()
data_also_view2 = lengths_also_view2.to_numpy()
data_also_buy2 = lengths_also_buy2.to_numpy()

# 박스플롯 그리기
plt.figure(figsize=(10, 6))
plt.boxplot([data_also_view, data_also_buy, data_also_view2, data_also_buy2], 
            labels=['cell (also_view)', 'cell (also_buy)', 'sport (also_view)', 'sport (also_buy)'])
plt.title('Box Plot of also_buy, also_view')
plt.xlabel('Category')
plt.ylabel('Length')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

def get_statistics(data):
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    minimum = np.min(data)
    maximum = np.max(data)
    mean = np.mean(data)
    return q1, q2, q3, minimum, maximum, mean

# 데이터 추출
data_also_view = lengths_also_view.to_numpy()
data_also_buy = lengths_also_buy.to_numpy()
data_also_view2 = lengths_also_view2.to_numpy()
data_also_buy2 = lengths_also_buy2.to_numpy()

# 요약 통계량 계산
statistics_also_view = get_statistics(data_also_view)
statistics_also_buy = get_statistics(data_also_buy)
statistics_also_view2 = get_statistics(data_also_view2)
statistics_also_buy2 = get_statistics(data_also_buy2)

# 박스플롯 그리기
plt.figure(figsize=(15, 10))
plt.boxplot([data_also_view, data_also_buy, data_also_view2, data_also_buy2], 
            labels=['cell (also_view)', 'cell (also_buy)', 'sport (also_view)', 'sport (also_buy)'],showmeans=True)

# 요약 통계량 표시
for i, statistics in enumerate([statistics_also_view, statistics_also_buy, statistics_also_view2, statistics_also_buy2]):
    plt.text(i + 1, statistics[3], f'Min: {statistics[3]}', ha='center', va='bottom', color='blue')
    plt.text(i + 1, statistics[0], f'Q1: {statistics[0]}', ha='center', va='bottom', color='blue')
    plt.text(i + 1, statistics[1], f'Median: {statistics[1]}', ha='center', va='bottom', color='blue')
    plt.text(i + 1, statistics[2], f'Q3: {statistics[2]}', ha='center', va='bottom', color='blue')
    plt.text(i + 1, statistics[4], f'Max: {statistics[4]}', ha='center', va='bottom', color='blue')
    plt.text(i + 1, statistics[5], f'Mean: {statistics[5]:.2f}', ha='center', va='bottom', color='red')

plt.title('Box Plot of also_buy, also_view')
plt.xlabel('Category')
plt.ylabel('Length')
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_ecdf(data):
    # 데이터를 정렬합니다.
    sorted_data = np.sort(data)
    # 각 데이터 값이 전체 데이터에서 차지하는 비율을 계산합니다.
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, ecdf

# 데이터 추출
data_also_view = lengths_also_view.to_numpy()
data_also_buy = lengths_also_buy.to_numpy()
data_also_view2 = lengths_also_view2.to_numpy()
data_also_buy2 = lengths_also_buy2.to_numpy()

# 누적분포 계산
sorted_also_view, ecdf_also_view = get_ecdf(data_also_view)
sorted_also_buy, ecdf_also_buy = get_ecdf(data_also_buy)
sorted_also_view2, ecdf_also_view2 = get_ecdf(data_also_view2)
sorted_also_buy2, ecdf_also_buy2 = get_ecdf(data_also_buy2)

# 누적분포 그래프 그리기
plt.figure(figsize=(15, 10))
plt.plot(sorted_also_view, ecdf_also_view, label='cell (also_view)')
plt.plot(sorted_also_buy, ecdf_also_buy, label='cell (also_buy)')
plt.plot(sorted_also_view2, ecdf_also_view2, label='sport (also_view)')
plt.plot(sorted_also_buy2, ecdf_also_buy2, label='sport (also_buy)')

plt.title('cum Plot of also_buy, also_view')
plt.xlabel('Length')
plt.ylabel('cum')
plt.grid(True)
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # 기초통계량 ex) 브랜드 유니크, 카테고리 유니크 -> 발표위한 것

# COMMAND ----------

# MAGIC %md
# MAGIC ## cellphone

# COMMAND ----------

print("cell brand:", len(cell_df["brand"].unique()))
print("cell cat2:", len(cell_df["cat2"].unique()))
print("cell cat3:", len(cell_df["cat3"].unique()))
print("cell cat4:", len(cell_df["cat4"].unique()))

# COMMAND ----------

display(cell_df)

# COMMAND ----------

sorted_counts = cell_df["cat2"].value_counts(dropna=False)
display(sorted_counts)

# COMMAND ----------

sorted_counts = cell_df["cat3"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------

sorted_counts = cell_df["cat4"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------

sorted_counts = cell_df["brand"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------

sorted_counts = cell_df["brand"].value_counts()
display(sorted_counts)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------

print("sport brand:", len(sport_df["brand"].unique()))
print("sport cat2:", len(sport_df["cat2"].unique()))
print("sport cat3:", len(sport_df["cat3"].unique()))
print("sport cat4:", len(sport_df["cat4"].unique()))

# COMMAND ----------

sorted_counts = sport_df["cat2"].value_counts(dropna=False)
display(sorted_counts)

# COMMAND ----------

sorted_counts = sport_df["cat3"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------

sorted_counts = sport_df["cat4"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------

sorted_counts = sport_df["brand"].value_counts().head(5)
display(sorted_counts)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 리뷰 데이터만 있는 asin, 메타만, 둘다, -> cell, sport 둘다 봐보기

# COMMAND ----------

# MAGIC %md
# MAGIC ## cellphone

# COMMAND ----------

# 테이블 읽기
cell_df = spark.read.table("asac.meta_cell_phones_and_accessories_new_price2")
review_cell = spark.read.table("asac.review_cellphone_accessories_final")

# pyspark pandas DataFrame으로 변경
cell_df = ps.DataFrame(cell_df)
review_cell = ps.DataFrame(review_cell)

# COMMAND ----------

display(cell_df.info())

# COMMAND ----------

display(review_cell.info())

# COMMAND ----------

len(cell_df["asin"].unique())

# COMMAND ----------

len(review_cell["asin"].unique())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM asac.review_cellphone_accessories_final as a
# MAGIC LEFT OUTER JOIN asac.meta_cell_phones_and_accessories_new_price2 as b ON a.asin = b.asin
# MAGIC
# MAGIC UNION
# MAGIC
# MAGIC SELECT *
# MAGIC FROM asac.review_cellphone_accessories_final as a
# MAGIC RIGHT OUTER JOIN asac.meta_cell_phones_and_accessories_new_price2 as b ON a.asin = b.asin

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VIEW joined_view AS
# MAGIC SELECT a.*, b.asin as b_asin, b.new_price2 
# MAGIC FROM asac.review_cellphone_accessories_final AS a
# MAGIC LEFT JOIN asac.meta_cell_phones_and_accessories_new_price2 AS b ON a.asin = b.asin
# MAGIC UNION
# MAGIC SELECT a.*, b.asin as b_asin, b.new_price2 
# MAGIC FROM asac.review_cellphone_accessories_final AS a
# MAGIC RIGHT JOIN asac.meta_cell_phones_and_accessories_new_price2 AS b ON a.asin = b.asin;

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from joined_view

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     COUNT(CASE WHEN asin IS NULL AND b_asin IS NOT NULL THEN 1 END) AS count_a_null_b_not_null,
# MAGIC     COUNT(CASE WHEN asin IS NOT NULL AND b_asin IS NULL THEN 1 END) AS count_a_not_null_b_null,
# MAGIC     COUNT(CASE WHEN asin IS NOT NULL AND b_asin IS NOT NULL THEN 1 END) AS count_a_not_null_b_not_null
# MAGIC FROM joined_view;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 조인 전 전체리뷰 : 10063255
# MAGIC #### 조인 후 전체 리뷰 : 10043502
# MAGIC
# MAGIC #### 리뷰x 메타o : 21
# MAGIC #### 리뷰o 메타x : 1699
# MAGIC #### 리뷰o 메타o : 10041789  -> 99.98%
# MAGIC
# MAGIC ##### -> 10,043,502 / 10,063,255 (99.8%) 
# MAGIC -> 10,063,255 - 10,043,502 = 19753

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from joined_view
# MAGIC where (asin is null) and (b_asin is null)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(*) from joined_view

# COMMAND ----------

# MAGIC %md
# MAGIC ## sport

# COMMAND ----------

# 테이블 읽기
sprot_df = spark.read.table("asac.sports_and_outdoors_fin_v2")
review_sport = spark.read.table("asac.reivew_sports_outdoor_final")

# pyspark pandas DataFrame으로 변경
sprot_df = ps.DataFrame(sprot_df)
review_sport = ps.DataFrame(review_sport)

# COMMAND ----------

display(sprot_df.info())

# COMMAND ----------

display(review_sport.info())

# COMMAND ----------

len(sport_df["asin"].unique())

# COMMAND ----------

len(review_sport["asin"].unique())

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VIEW joined_view_sport AS
# MAGIC SELECT a.*, b.asin as b_asin, b.new_price3
# MAGIC FROM asac.reivew_sports_outdoor_final AS a
# MAGIC LEFT OUTER JOIN asac.sports_and_outdoors_fin_v2 AS b ON a.asin = b.asin
# MAGIC UNION
# MAGIC SELECT a.*, b.asin as b_asin, b.new_price3 
# MAGIC FROM asac.reivew_sports_outdoor_final AS a
# MAGIC RIGHT OUTER JOIN asac.sports_and_outdoors_fin_v2 AS b ON a.asin = b.asin;

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from joined_view_sport

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(*) from joined_view_sport

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     COUNT(CASE WHEN asin IS NULL AND b_asin IS NOT NULL THEN 1 END) AS count_a_null_b_not_null,
# MAGIC     COUNT(CASE WHEN asin IS NOT NULL AND b_asin IS NULL THEN 1 END) AS count_a_not_null_b_null,
# MAGIC     COUNT(CASE WHEN asin IS NOT NULL AND b_asin IS NOT NULL THEN 1 END) AS count_a_not_null_b_not_null
# MAGIC FROM joined_view_sport;

# COMMAND ----------

# MAGIC %md
# MAGIC #### 조인 전 전체리뷰 : 12980837
# MAGIC #### 조인 후 전체 리뷰 : 12622891
# MAGIC
# MAGIC #### 리뷰x 메타o : 25
# MAGIC #### 리뷰o 메타x : 13388
# MAGIC #### 리뷰o 메타o : 12609478 -> 중복제거후 99.89%
# MAGIC ##### -> 12,622,891 / 12,980,837 (97.24%)
# MAGIC -> 12,980,837 - 12,622,891 = 357,946

# COMMAND ----------


