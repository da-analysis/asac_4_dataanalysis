# Databricks notebook source
# MAGIC %sql
# MAGIC select *
# MAGIC from asac.`240510_new_train_prediction`
# MAGIC limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC select asin1, asin2, ROW_NUMBER() OVER (PARTITION BY asin1 ORDER BY prob asc, review_cnts asc, total_review_counts asc, average_overall asc) AS score
# MAGIC                    , RANK() OVER (PARTITION BY asin1 ORDER BY prob desc, review_cnts desc, total_review_counts desc, average_overall desc) AS rank
# MAGIC from asac.`240510_new_train_prediction`
# MAGIC where asin1 = 'B00UV70YVC'

# COMMAND ----------

prediction_rank = spark.sql("""
          select *, ROW_NUMBER() OVER (PARTITION BY asin1 ORDER BY prob asc, review_cnts asc, total_review_counts asc, average_overall asc) AS score
                  , RANK() OVER (PARTITION BY asin1 ORDER BY prob desc, review_cnts desc, total_review_counts desc, average_overall desc) AS rank
          from asac.`240510_new_train_prediction`
""")

# COMMAND ----------

name = "asac.`240510_new_train_prediction_rank`"
prediction_rank.write.saveAsTable(name)

# COMMAND ----------

recommend_df = spark.sql("""
select asin1, asin2, cast(score as float) as relevance
from asac.`240510_new_train_prediction_rank`
""")

# COMMAND ----------

test_df = spark.sql("""
select asin1, asin2, cast(review_cnts as float) as relevance
from asac.`240430_test_df`
""")

# COMMAND ----------

from replay.metrics import Coverage, HitRate, NDCG, MAP, OfflineMetrics, Precision, Recall, Surprisal, MRR
from replay.metrics.experiment import Experiment

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.data.dataset_utils import DatasetLabelEncoder

# COMMAND ----------

K=10

# COMMAND ----------

ex = Experiment([MAP(K),
                 NDCG([K]),
                 HitRate([K]),
                 Coverage(K),
                 MRR([K]),
                 Precision([K]),
                 Recall([K])], 
                ground_truth = test_df, 
                train = recommend_df,
                base_recommendations = recommend_df,
                query_column = 'asin1', 
                item_column = 'asin2', 
                rating_column='relevance') 

# COMMAND ----------

ex.add_result("baseline", recommend_df)

# COMMAND ----------

display(ex.results)

# COMMAND ----------

pred_6 = spark.sql("""
select asin1, asin2, cast(prob as float) as relevance
from asac.pred_6
""")

pred_1 = spark.sql("""
select asin1, asin2, cast(prob as float) as relevance
from asac.pred_1_we
""")

# COMMAND ----------

ex.add_result("text model", pred_1)

# COMMAND ----------

ex.add_result("text&image model", pred_6)

# COMMAND ----------

ex.results

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### threshold 방식 (이건 안쓸 것임)

# COMMAND ----------

pred_6_05 = spark.sql("""
select asin1, asin2, cast(prob as float) as relevance
from asac.pred_6
where prob >= 0.5
""")

pred_1_05 = spark.sql("""
select asin1, asin2, cast(prob as float) as relevance
from asac.pred_1_we
where prob >= 0.5
""")

# COMMAND ----------

ex.add_result("text model (threshold>=0.5)", pred_1_05)

# COMMAND ----------

ex.add_result("text&image model (threshold>=0.5)", pred_6_05)

# COMMAND ----------

ex.results

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

test_df_review2 = spark.sql("""
select asin1, asin2, cast(review_cnts as float) as relevance
from asac.`240430_test_df`
where review_cnts >= 2
""")

# COMMAND ----------

ex2 = Experiment([MAP(K),
                 NDCG([K]),
                 HitRate([1,K]),
                 Coverage(K),
                 MRR([K]),
                 Precision([K]),
                 Recall([K])], 
                ground_truth = test_df_review2, 
                train = recommend_df,
                base_recommendations = recommend_df,
                query_column = 'asin1', 
                item_column = 'asin2', 
                rating_column='relevance') 

# COMMAND ----------

ex2.add_result("baseline", recommend_df)

# COMMAND ----------

ex2.add_result("text model", pred_1)

# COMMAND ----------

ex2.add_result("text&image model", pred_6)

# COMMAND ----------

ex2.results
