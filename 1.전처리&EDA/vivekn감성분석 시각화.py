# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`asac`.`senti_vivekn_fin`;

# COMMAND ----------

from pyspark.sql.functions import length

df= _sqldf.withColumn('reviewTextLength', length(_sqldf['reviewText']))

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `hive_metastore`.`asac`.`senti_trans_1000`;

# COMMAND ----------

df = _sqldf

# COMMAND ----------

display(df)

# COMMAND ----------


