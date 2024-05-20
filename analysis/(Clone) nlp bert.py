# Databricks notebook source
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings, AlbertEmbeddings, Tokenizer, Normalizer, StopWordsCleaner, RoBertaSentenceEmbeddings
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, DoubleType
import numpy as np
from pyspark.sql.functions import col, size, array, expr
dot_udf = F.udf(lambda x,y: float(x.dot(y)) / float(x.norm(2)*y.norm(2)), DoubleType())

import operator
import datetime as dt
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel, Normalizer
from pyspark.sql.functions import format_number as fmt
from pyspark.sql.types import FloatType, DoubleType
import numpy as np
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from pyspark.ml.linalg import Vectors

from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType
dot_udf = F.udf(lambda x,y: float(x.dot(y)) / float(x.norm(2)*y.norm(2)), DoubleType())

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cosine_similarity_udf = udf(cosine_similarity, FloatType())


import pyspark.sql.functions as F
# define udf
def sorter(l):
  res = sorted(l, key=operator.itemgetter(0))
  return [item[1] for item in res]

sort_udf = F.udf(sorter)

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
    .setOutputCol("token")
  
embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128", "en") \
    .setInputCols(["sentences"]) \
    .setOutputCol("sentence_bert_embeddings")\
    .setCaseSensitive(True) \
    .setMaxSentenceLength(512)

# embeddings = AlbertEmbeddings.pretrained("albert_base_uncased", "en") \
# .setInputCols("sentences", "token") \
# .setOutputCol("embeddings")

# COMMAND ----------

nlp_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, embeddings])

# COMMAND ----------

text_df = spark.sql(""" select * 
                        from default.all_amazon_data 
                        where cat_1 = 'Clothing, Shoes & Jewelry' 
""")  # and a.cat_1 = 'Clothing, Shoes & Jewelry'

# COMMAND ----------

model = nlp_pipeline.fit(text_df)
result_embeddings = model.transform(text_df)

# COMMAND ----------

display(result_embeddings)

# COMMAND ----------

result_embeddings.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/nlp/sent_small_bert_L2_128')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

sentence_bert = spark.read.parquet('dbfs:/FileStore/amazon/data/nlp/sent_small_bert_L2_128')

# COMMAND ----------

sentence_bert.count()

# COMMAND ----------

display(sentence_bert)

# COMMAND ----------

display(result_df)

# COMMAND ----------

from sparknlp.base import LightPipeline
light_model_emb = LightPipeline(pipelineModel = model, parse_embeddings=True)

# COMMAND ----------

text_df

# COMMAND ----------

annotate_results_emb = light_model_emb.annotate('Good spit from Rich Homie Quan.')

# COMMAND ----------

list(zip(annotate_results_emb['sentence'], annotate_results_emb['sentence_bert_embeddings']))

# COMMAND ----------

sentence_bert = spark.read.parquet('dbfs:/FileStore/amazon/data/nlp/sent_small_bert_L2_128')
sentence_bert.createOrReplaceTempView("sentence_bert")

# COMMAND ----------

# 여러 배열의 평균을 계산하는 함수
def average_of_arrays(*arrays):
    if len(arrays) == 0:
        return []
    
    array_length = len(arrays[0])
    for arr in arrays:
        if len(arr) != array_length:
            raise ValueError("All arrays must have the same length")
    
    return [sum(arr[i] for arr in arrays) / len(arrays) for i in range(array_length)]
# UDF로 등록
average_of_arrays_udf = udf(average_of_arrays, ArrayType(DoubleType()))

# 데이터프레임 예시
df = spark.createDataFrame([([1, 2, 3], [4, 5, 6],), ([7, 8, 9], [10, 11, 12],)], ['array1', 'array2'])

# COMMAND ----------

df.withColumn("average", average_of_arrays_udf(df["array1"], df["array2"])).show()

# COMMAND ----------

# 예제 데이터
data = [("ASIN1", [1, 2, 3]), ("ASIN1", [4, 5, 6]), ("ASIN2", [7, 8, 9]), ("ASIN2", [10, 11, 12])]
df = spark.createDataFrame(data, ["asin", "scores"])

# 배열 길이 계산
array_length = 128

# 배열의 평균을 계산하는 함수
def average_arrays(arrays):
    sum_arrays = [0] * array_length
    for arr in arrays:
        sum_arrays = [sum(x) for x in zip(sum_arrays, arr + [0]*(array_length - len(arr)))]
    return [x / len(arrays) for x in sum_arrays]

# UDF 등록
spark.udf.register("average_arrays_udf", average_arrays, ArrayType(DoubleType()))

# 각 ASIN별로 배열의 평균 계산
result = df.groupBy("asin").agg(expr("average_arrays_udf(collect_list(scores))").alias("average_scores"))

# COMMAND ----------

display(result)

# COMMAND ----------

sentence_bert_asin = spark.sql("""
select t.*, sentence_bert_embeddings.begin as sbe_begin
              , sentence_bert_embeddings.end as sbe_end
              , sentence_bert_embeddings.result as sbe_result
              , sentence_bert_embeddings.metadata.sentence as sbe_metadata_sentence
              , sentence_bert_embeddings.embeddings as sbe_embeddings
from(
    select asin, date_column, reviewerID, overall, vote_int, date_column, reviewText, summary, sentences, explode(sentence_bert_embeddings) sentence_bert_embeddings
    from sentence_bert 
    where 1=1
    --- and asin = 'B0060EY8GC'
    and date_column < '2018-01-01'
) as t
""") 

sentence_bert_asin.createOrReplaceTempView("sentence_bert_asin")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct asin)
# MAGIC from sentence_bert

# COMMAND ----------

# display(sentence_bert_asin)

# COMMAND ----------

sentence_bert_asin = sentence_bert_asin.groupBy("asin").agg(expr("average_arrays_udf(collect_list(sbe_embeddings))").alias("average_scores"))

# COMMAND ----------

# display(sentence_bert_asin)

# COMMAND ----------

sentence_bert_asin.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/nlp/asin/sent_small_bert_L2_128_average')

# COMMAND ----------

bert_df = spark.read.parquet('dbfs:/FileStore/amazon/data/nlp/asin/sent_small_bert_L2_128_average')
bert_df.createOrReplaceTempView("bert_df")

# COMMAND ----------

cooccurrences_df = spark.read.parquet("dbfs:/FileStore/amazon/data/recommend_data/cooccurrences")
cooccurrences_df.createOrReplaceTempView("cooccurrences_df")

train_df = spark.sql("""
select asin1, asin2, sum(review_cnts) review_cnts, sum(review_5_cnts) review_5_cnts, sum(review_4more_cnts) review_4more_cnts
from cooccurrences_df
where date_column < '2018-01-01' and date_add(date_column, -date_diff) < '2018-01-01'
group by asin1, asin2
""")

test_df = spark.sql("""
select asin1, asin2, sum(review_cnts) review_cnts, sum(review_5_cnts) review_5_cnts, sum(review_4more_cnts) review_4more_cnts
from cooccurrences_df
where date_column >= '2018-01-01' or date_add(date_column, -date_diff) >= '2018-01-01'
group by asin1, asin2
""")

train_df.createOrReplaceTempView("train_df")
test_df.createOrReplaceTempView("test_df")

# COMMAND ----------

train_df2 = spark.sql("""
select a.*, b.average_scores as asin1_vector, c.average_scores as asin2_vector
from train_df as a 
inner join bert_df as b on 1=1 and a.asin1 = b.asin
inner join bert_df as c on 1=1 and a.asin2 = c.asin
""")

train_df2 = train_df2.withColumn('dot',   F.expr('aggregate(arrays_zip(asin1_vector, asin2_vector), 0D, (acc, x) -> acc + (x.asin1_vector * x.asin2_vector))')) \
  .withColumn('norm1', F.expr('sqrt(aggregate(asin1_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('norm2', F.expr('sqrt(aggregate(asin2_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('cos_vector', F.expr('dot / (norm1 * norm2)'))

# COMMAND ----------

train_df2.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/recommend_data/sent_small_bert_L2_128_average')

# COMMAND ----------

bert_cf = spark.read.parquet('dbfs:/FileStore/amazon/data/recommend_data/sent_small_bert_L2_128_average')

# COMMAND ----------

bert_cf.createOrReplaceTempView("bert_cf")

# COMMAND ----------

bert_cf = bert_cf.withColumn('dot',   F.expr('aggregate(arrays_zip(asin1_vector, asin2_vector), 0D, (acc, x) -> acc + (x.asin1_vector * x.asin2_vector))')) \
  .withColumn('norm1', F.expr('sqrt(aggregate(asin1_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('norm2', F.expr('sqrt(aggregate(asin2_vector, 0D, (acc, x) -> acc + (x * x)))')) \
  .withColumn('cos_vector', F.expr('dot / (norm1 * norm2)'))

# COMMAND ----------

# bert_cf = bert_cf.withColumn("cos_vector", F.when((F.col('asin1_vector').isNull()) | (F.col('asin2_vector').isNull()), 0).otherwise(dot_udf(F.col('asin1_vector'), F.col('asin2_vector'))))
# bert_cf = bert_cf.withColumn("cos_vector", F.when((F.col('asin1_vector').isNull()) | (F.col('asin2_vector').isNull()), 0).otherwise(cosine_similarity_udf(F.col('asin1_vector'), F.col('asin2_vector'))))

# COMMAND ----------

label_dt= spark.read.parquet('dbfs:/FileStore/amazon/data/recommend_data/cooccurrences/test')
label_dt.createOrReplaceTempView("label_dt")

label_dt = spark.sql("""
select t.*
from(                     
    select asin1, asin2, sum(review_cnts) review_cnts, sum(review_5_cnts) review_5_cnts, sum(review_4more_cnts) review_4more_cnts
    from label_dt 
    group by asin1, asin2
) as t
where 1=1
and t.review_4more_cnts >= 2
""")
label_dt.createOrReplaceTempView("label_dt")

# COMMAND ----------

def evaluation_function(prediction, label_df, col_1, col_2, col_rt, col_pre, reco_k, modelname, relevancy_method,  threshold):
    start = dt.datetime.now()
    
    prediction.createOrReplaceTempView("prediction")  
    
    spark.sql("""
        select count(distinct {0}) col_1_cnts, count(distinct {1}) col_2_cnts, count(*) cnts
        from prediction
    """.format(col_1, col_2)).show()
    
    label_df.createOrReplaceTempView("label_df")  
    
    spark.sql("""
        select count(distinct {0}) col_1_cnts, count(distinct {1}) col_2_cnts, count(*) cnts
        from label_df
    """.format(col_1, col_2)).show()

    if relevancy_method == 'top_k':
        spark.sql("""
            select count(distinct a.{0}) col_1_cnts, count(distinct a.{1}) col_2_cnts, count(*) cnts
            from prediction as a
            inner join label_df as b
            on 1=1
            and a.{0} = b.{0}
            and a.{1} = b.{1}
        """.format(col_1, col_2)).show()   
    elif relevancy_method == 'by_threshold':
         spark.sql("""
            select count(distinct a.{0}) col_1_cnts, count(distinct a.{1}) col_2_cnts, count(*) cnts
            from prediction as a
            inner join label_df as b
            on 1=1
            and a.{0} = b.{0}
            and a.{1} = b.{1}
            where 1=1
            and a.{2} >= {3}
        """.format(col_1, col_2, col_pre, threshold)).show()         
    else:
        spark.sql("""
            select count(distinct a.{0}) col_1_cnts, count(distinct a.{1}) col_2_cnts, count(*) cnts
            from prediction as a
            inner join label_df as b
            on 1=1
            and a.{0} = b.{0}
            and a.{1} = b.{1}
        """.format(col_1, col_2)).show()   

    ### Recommend Model Performance     
    rank_eval = SparkRankingEvaluation(label_df, prediction, k = reco_k, col_user=col_1, col_item=col_2, 
                                    col_rating=col_rt, col_prediction=col_pre, 
                                    relevancy_method=relevancy_method, threshold=threshold)
                                    
    print("Model: {}".format(modelname), "Top K:%d" % rank_eval.k, "MAP:%f" % rank_eval.map_at_k(), "NDCG:%f" % rank_eval.ndcg_at_k(), "Precision@K:%f" % rank_eval.precision_at_k(), "Recall@K:%f" % rank_eval.recall_at_k())
    
    end = dt.datetime.now() - start
    print (end)
    
    return(rank_eval)

# COMMAND ----------

rank_eval_meta = evaluation_function(
                       prediction = bert_cf, 
                       label_df = label_dt, 
                       col_1 = "asin1", 
                       col_2 = "asin2", 
                       col_rt = "review_4more_cnts", 
                       col_pre = "cos_vector",
                       reco_k = 10, 
                       modelname = "bert_cf", 
                       relevancy_method = 'top_k',  
                       threshold = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### ELECTRA Sentence

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

embeddings = BertSentenceEmbeddings.pretrained("sent_electra_base_uncased", "en") \
    .setInputCols(["sentences"]) \
    .setOutputCol("sentence_bert_embeddings")\
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

nlp_electra_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, embeddings])

# COMMAND ----------

text_df = spark.sql(""" select * 
                        from default.all_amazon_data 
                        where cat_1 = 'Clothing, Shoes & Jewelry' 
""")  # and a.cat_1 = 'Clothing, Shoes & Jewelry'

# COMMAND ----------

model = nlp_electra_pipeline.fit(text_df)
result_embeddings = model.transform(text_df)

# COMMAND ----------

result_embeddings.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/nlp/sent_electra_base_uncased')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Robert

# COMMAND ----------

documentAssembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

embeddings = RoBertaSentenceEmbeddings.pretrained("sent_roberta_base", "en") \
    .setInputCols(["sentences"]) \
    .setOutputCol("sentence_bert_embeddings")\
    .setCaseSensitive(True) \
    .setMaxSentenceLength(128)

nlp_roberta_pipeline = Pipeline(stages=[documentAssembler, sentenceDetector, embeddings])

# COMMAND ----------

model = nlp_roberta_pipeline.fit(text_df)
result_embeddings = model.transform(text_df)

# COMMAND ----------

result_embeddings.write.mode("overwrite").parquet('dbfs:/FileStore/amazon/data/nlp/sent_roberta_base')

# COMMAND ----------

# https://sparknlp.org/2021/09/01/sent_roberta_base_en.html
