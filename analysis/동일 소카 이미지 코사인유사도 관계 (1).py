# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`asac`.`image_model`;

# COMMAND ----------

df = _sqldf

# COMMAND ----------

def compute_same_category_ratio(df, lower_bound, upper_bound):
    relevant_count = df.filter(
        (col("cosine_similarity") >= lower_bound) & 
        (col("cosine_similarity") < upper_bound) & 
        (col("cat3_equal") == 1)
    ).count()
    
    total_count = df.filter(
        (col("cosine_similarity") >= lower_bound) & 
        (col("cosine_similarity") < upper_bound)
    ).count()
    
    ratio = relevant_count / total_count if total_count > 0 else 0
    return ratio

# 코사인 유사도 구간별 동일 카테고리 비율 계산
intervals = [
    (0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6),
    (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1),
    (-0.1, 0.0), (-0.2, -0.1), (-0.3, -0.2), (-0.4, -0.3), (-0.5, -0.4),
    (-0.6, -0.5), (-0.7, -0.6), (-0.8, -0.7), (-0.9, -0.8), (-1.0, -0.9)
]

for lower, upper in intervals:
    ratio = compute_same_category_ratio(df, lower, upper)
    print(f"코사인 유사도 {lower} 이상 {upper} 미만: {ratio}")

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col

def compute_same_category_ratio(df, lower_bound, upper_bound):
    relevant_count = df.filter(
        (col("cosine_similarity") >= lower_bound) & 
        (col("cosine_similarity") < upper_bound) & 
        (col("cat3_equal") == 1)
    ).count()
    
    total_count = df.filter(
        (col("cosine_similarity") >= lower_bound) & 
        (col("cosine_similarity") < upper_bound)
    ).count()
    
    ratio = relevant_count / total_count if total_count > 0 else 0
    return ratio

# 코사인 유사도 구간별 동일 카테고리 비율 계산
intervals = [
    (0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6),
    (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1),
    (-0.1, 0.0), (-0.2, -0.1), (-0.3, -0.2), (-0.4, -0.3), (-0.5, -0.4),
    (-0.6, -0.5), (-0.7, -0.6), (-0.8, -0.7), (-0.9, -0.8), (-1.0, -0.9)
]

ratios = []
for lower, upper in intervals:
    ratio = compute_same_category_ratio(df, lower, upper)
    ratios.append(ratio)

# 시각화
labels = [f"{lower} ~ {upper}" for lower, upper in intervals]
plt.figure(figsize=(12, 6))
plt.bar(labels, ratios)
plt.xlabel("Cosine Similarity Interval")
plt.ylabel("Same Category Ratio")
plt.title("")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# COMMAND ----------


