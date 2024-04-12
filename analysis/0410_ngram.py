# Databricks notebook source
# MAGIC %md
# MAGIC ## Ngram & WordCloud

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
%pip install --upgrade pillow
%pip install --upgrade pip

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import matplotlib.pyplot as plt

# 테이블 읽기
cell_meta = spark.read.table("asac.meta_cell_phones_and_accessories_new_price2")
sport_meta = spark.read.table("asac.sports_and_outdoors_fin_v2")
cell_review = spark.read.table("asac.review_cellphone_accessories_final")
sport_review = spark.read.table("asac.reivew_sports_outdoor_final")


# pyspark pandas DataFrame으로 변경
cell_meta = ps.DataFrame(cell_meta)
sport_meta = ps.DataFrame(sport_meta)
cell_review = ps.DataFrame(cell_review)
sport_review = ps.DataFrame(sport_review)

# COMMAND ----------

# reviewText, asin, reviewerID 만 뽑아서 테이블 합치기
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

cell_review_text = spark.read.table("asac.review_cellphone_accessories_final").select("reviewText","asin","reviewerID")
sport_review_text = spark.read.table("asac.reivew_sports_outdoor_final").select("reviewText","asin","reviewerID")

review_text = cell_review_text.union(sport_review_text)

review_text.show(truncate=False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline 구축

# COMMAND ----------

from sparknlp.annotator import Lemmatizer, Stemmer, Tokenizer, Normalizer, LemmatizerModel
from sparknlp.annotator import Chunker
from sparknlp.base import Finisher, EmbeddingsFinisher
from nltk.corpus import stopwords
from sparknlp.annotator import StopWordsCleaner
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import PerceptronModel     
from sparknlp.annotator import Chunker
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from sparknlp.annotator import (
    SentenceDetector,
    Normalizer,
    StopWordsCleaner,
    Doc2VecModel
)
import nltk
nltk.download('stopwords')

from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC **품사 태깅 종류**
# MAGIC <img src="https://mblogthumb-phinf.pstatic.net/MjAyMDA0MDZfMTUz/MDAxNTg2MTQzOTE2MDc3._q5jz1Y50qyH23mv5VsU_Vz_s6At_CnVQl-HbyL873wg.Bk1q6ZSCUOJ5rxy5yZGBKTaBpnVbnPdvu_A3a1vyzfEg.PNG.bycho211/1.png?type=w800">

# COMMAND ----------

from pyspark.sql.functions import regexp_replace
# 태그 제거
cell_review_text = cell_review_text.withColumn("no_tag", regexp_replace("reviewText", "[\\n\\t\/<>]", " "))
sport_review_text = sport_review_text.withColumn("no_tag", regexp_replace("reviewText", "[\\n\\t\/<>]", " "))

# 문장부호 제거
cell_review_text = cell_review_text.withColumn("reviewText_nofunc", regexp_replace("no_tag", r"[!?,.\(\)\[\]\{\}_:;#@=\"]", " "))
sport_review_text = sport_review_text.withColumn("reviewText_nofunc", regexp_replace("no_tag", r"[!?,.\(\)\[\]\{\}_:;#@]", " "))

# 소문자 통일
from pyspark.sql.functions import lower

# reviewText_nofunc 열의 문자열을 소문자로 변환
cell_review_text = cell_review_text.withColumn("reviewText_nofunc_lower", lower("reviewText_nofunc"))
sport_review_text = sport_review_text.withColumn("reviewText_nofunc_lower", lower("reviewText_nofunc"))

display(cell_review_text.head(20))

# COMMAND ----------

# 태그제거, 문장부호제거, 토크나이즈, 불용어 제거, 표제어 추출 + (+태그제거 + 특수 문자 추가로 제거함 + 소문자 통일)
documentAssembler = DocumentAssembler().setInputCol("reviewText_nofunc_lower").setOutputCol("document") # 원시 데이터를 문서 형태로 변환
tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('tokenized') # 토큰화

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

lemmatizer = LemmatizerModel.pretrained()\
     .setInputCols(['normalized'])\
     .setOutputCol('lemmatized') # 표제어 추출 (어근추출과는 살짝 다름)

eng_stopwords = stopwords.words('english')

stopwords_cleaner = StopWordsCleaner()\
     .setInputCols(['lemmatized'])\
     .setOutputCol('no_stop_lemmatized')\
     .setStopWords(eng_stopwords) # 불용어 제거

pos_tagger = PerceptronModel.pretrained('pos_anc') \
     .setInputCols(['document', 'lemmatized']) \
     .setOutputCol('pos') # 품사 태깅

allowed_2gram_tags = ['<JJ>+<NN>', '<NN>+<NN>']
chunker_2 = Chunker() \
      .setInputCols(['document', 'pos']) \
      .setOutputCol('ngrams') \
      .setRegexParsers(allowed_2gram_tags)


allowed_3gram_tags = ['<RB>+<JJ>+<NN>', '<NN>+<NN>+<RB>', '<JJ>+<NN>+<NN>'] # RB : adverb, JJ : adjective, NN : noun
chunker_3 = Chunker() \
     .setInputCols(['document', 'pos']) \
     .setOutputCol('ngrams') \
     .setRegexParsers(allowed_3gram_tags) # 조건과 일치하는 품사 조합

allowed_4gram_tags = ['<RB>+<RB>+<JJ>+<NN>'] #'<RB>+<JJ>+<NN>+<NN>', 
chunker_4 = Chunker() \
     .setInputCols(['document', 'pos']) \
     .setOutputCol('ngrams') \
     .setRegexParsers(allowed_4gram_tags) 


finisher = Finisher() \
     .setInputCols(['ngrams']) # 결과를 string으로 출력 


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### cellphone과 sport 나눠서 2gram 분석
# MAGIC - cell_review_text
# MAGIC - sport_review_text

# COMMAND ----------

pipeline_2gram = Pipeline() \
     .setStages([documentAssembler,
                 tokenizer,
                 normalizer,
                 lemmatizer,
                 stopwords_cleaner,
                 pos_tagger,
                 chunker_2,
                 finisher])

# COMMAND ----------

#pipeline_2gram = Pipeline() \
#     .setStages([documentAssembler,
#                 tokenizer,
#                 normalizer,
#                 lemmatizer,
#                 stopwords_cleaner,
#                 pos_tagger,
#                 chunker_2])

# COMMAND ----------

# 임시 뷰 생성
cell_review_text.createOrReplaceTempView("cell_review_text")
sport_review_text.createOrReplaceTempView("sport_review_text")

# COMMAND ----------

processed_cell = pipeline_2gram.fit(cell_review_text).transform(cell_review_text)
processed_sport = pipeline_2gram.fit(sport_review_text).transform(sport_review_text)

# COMMAND ----------

display(processed_cell)

# COMMAND ----------

display(processed_sport)

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import explode, concat_ws, collect_list

# 2grams 열을 explode하여 각 단어를 개별 행으로 변환
exploded_df_cell = processed_cell.withColumn("2gram", explode("finished_ngrams"))
exploded_df_sport = processed_sport.withColumn("2gram", explode("finished_ngrams"))

# COMMAND ----------

display(exploded_df_cell)

# COMMAND ----------

display(exploded_df_sport)

# COMMAND ----------

from pyspark.sql.functions import col
cell_10000 = exploded_df_cell.limit(10000)
result_cell_2gram = cell_10000.groupBy("2gram").count()
result_cell_2gram.show(10)

# COMMAND ----------

result_dict = {}

for row in result_cell_2gram.collect():
    key = row["2gram"]
    value = row["count"]
    result_dict[key] = value

print(result_dict)

# COMMAND ----------

from wordcloud import WordCloud
wc2_cell = WordCloud(background_color='white', max_words=50, min_font_size=6, colormap='Dark2')

fig = plt.figure(figsize=(20, 20))
wc2_cell = wc2_cell.generate_from_frequencies(result_dict)
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 2-gram')
plt.imshow(wc2_cell)
plt.axis('off')

# COMMAND ----------

from pyspark.sql.functions import col
sport_10000 = exploded_df_sport.limit(10000)
result_sport_2gram = sport_10000.groupBy("2gram").count()
result_sport_2gram.show(10)

# COMMAND ----------

result_dict_sport = {}

for row in result_sport_2gram.collect():
    key = row["2gram"]
    value = row["count"]
    result_dict_sport[key] = value

print(result_dict_sport)

# COMMAND ----------

from wordcloud import WordCloud
wc2_cell = WordCloud(background_color='black', max_words=50, min_font_size=6, colormap='Pastel2')
wc2_sport = WordCloud(background_color='white', max_words=50, min_font_size=6, colormap='Dark2')

fig = plt.figure(figsize=(20, 20))
wc2_cell = wc2_cell.generate_from_frequencies(result_dict)
wc2_sport = wc2_sport.generate_from_frequencies(result_dict_sport)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 2-gram')
plt.imshow(wc2_cell)
plt.axis('off')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('sport 2-gram')
plt.imshow(wc2_sport)
plt.axis('off')
plt.show()

# COMMAND ----------

result_cell_2gram = exploded_df_cell.groupBy("2gram").count()
result_sport_2gram = exploded_df_cell.groupBy("2gram").count()

# COMMAND ----------

fre2_cell = result_cell_2gram.set_index('2gram').to_dict()['count']
fre2_sport = result_sport_2gram.set_index('2gram').to_dict()['count']

# COMMAND ----------

wc2_cell = WordCloud(background_color='white', max_words=100, min_font_size=6, colormap='Dark2')
wc2_sport = WordCloud(background_color='white', max_words=100, min_font_size=6, colormap='Dark2')

fig = plt.figure(figsize=(20, 20))
wc2_cell = wc2_cell.generate_from_frequencies(fre2_cell)
wc2_sport = wc2_sport.generate_from_frequencies(fre2_sport)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 2-gram')
plt.imshow(wc2_cell)
plt.axis('off')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('sport 2-gram')
plt.imshow(wc2_sport)
plt.axis('off')
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### cellphone과 sport 나눠서 3gram 분석
# MAGIC - cell_review_text
# MAGIC - sport_review_text

# COMMAND ----------

pipeline_3gram = Pipeline() \
     .setStages([documentAssembler,
                 tokenizer,
                 normalizer,
                 lemmatizer,
                 stopwords_cleaner,
                 pos_tagger,
                 chunker_3,
                 finisher])

# COMMAND ----------

processed_3gram_cell = pipeline_3gram.fit(cell_review_text).transform(cell_review_text)
processed_3gram_sport = pipeline_3gram.fit(sport_review_text).transform(sport_review_text)

# COMMAND ----------

display(processed_3gram_cell)

# COMMAND ----------

display(processed_3gram_sport)

# COMMAND ----------

from pyspark.sql.functions import explode, concat_ws, collect_list

# 3grams 열을 explode하여 각 단어를 개별 행으로 변환
exploded_cell_3gram = processed_3gram_cell.withColumn("3gram", explode("finished_ngrams"))
exploded_sport_3gram = processed_3gram_sport.withColumn("3gram", explode("finished_ngrams"))

# COMMAND ----------

display(exploded_cell_3gram)

# COMMAND ----------

display(exploded_sport_3gram)

# COMMAND ----------

from pyspark.sql.functions import col
exploded_cell_3gram = exploded_cell_3gram.limit(10000)
result_cell_3gram = exploded_cell_3gram.groupBy("3gram").count()

# display(result_cell_3gram.head(10))

# COMMAND ----------

exploded_sport_3gram= exploded_sport_3gram.limit(10000)
result_sport_3gram = exploded_sport_3gram.groupBy("3gram").count()

#display(result_sport_3gram.head(10))

# COMMAND ----------

result_dict_cell_3gram = {}

for row in result_cell_3gram.collect():
    key = row["3gram"]
    value = row["count"]
    result_dict_cell_3gram[key] = value

#print(result_dict_cell_3gram)

# COMMAND ----------

result_dict_sport_3gram = {}

for row in result_sport_3gram.collect():
    key = row["3gram"]
    value = row["count"]
    result_dict_sport_3gram[key] = value

# print(result_dict_sport_3gram)

# COMMAND ----------

wc3_cell = WordCloud(background_color='black', max_words=50, min_font_size=8, colormap='Pastel1')
wc3_sport = WordCloud(background_color='white', max_words=50, min_font_size=8, colormap='rainbow_r')

fig = plt.figure(figsize=(20, 20))
wc3_cell = wc3_cell.generate_from_frequencies(result_dict_cell_3gram)
wc3_sport = wc3_sport.generate_from_frequencies(result_dict_sport_3gram)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 3-gram')
plt.imshow(wc3_cell)
plt.axis('off')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('sport 3-gram')
plt.imshow(wc3_sport)
plt.axis('off')
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### cellphone과 sport 나눠서 4gram 분석

# COMMAND ----------

pipeline_4gram = Pipeline() \
     .setStages([documentAssembler,
                 tokenizer,
                 normalizer,
                 lemmatizer,
                 stopwords_cleaner,
                 pos_tagger,
                 chunker_4,
                 finisher])

# COMMAND ----------

processed_4gram_cell = pipeline_4gram.fit(cell_review_text).transform(cell_review_text)
processed_4gram_sport = pipeline_4gram.fit(sport_review_text).transform(sport_review_text)

# COMMAND ----------

display(processed_4gram_cell)

# COMMAND ----------

display(processed_4gram_sport)

# COMMAND ----------

from pyspark.sql.functions import explode, concat_ws, collect_list

# 4grams 열을 explode하여 각 단어를 개별 행으로 변환
exploded_cell_4gram = processed_4gram_cell.withColumn("4gram", explode("finished_ngrams"))
exploded_sport_4gram = processed_4gram_sport.withColumn("4gram", explode("finished_ngrams"))

# COMMAND ----------

display(exploded_cell_4gram)

# COMMAND ----------

display(exploded_sport_4gram)

# COMMAND ----------

from pyspark.sql.functions import col
exploded_cell_4gram_10000 = exploded_cell_4gram.limit(10000)
result_cell_4gram = exploded_cell_4gram_10000.groupBy("4gram").count()

# COMMAND ----------

exploded_sport_4gram_10000 = exploded_sport_4gram.limit(10000)
result_sport_4gram = exploded_sport_4gram_10000.groupBy("4gram").count()

# COMMAND ----------

result_dict_cell_4gram = {}

for row in result_cell_4gram.collect():
    key = row["4gram"]
    value = row["count"]
    result_dict_cell_4gram[key] = value

#print(result_dict_cell_4gram)

# COMMAND ----------

result_dict_sport_4gram = {}

for row in result_sport_4gram.collect():
    key = row["4gram"]
    value = row["count"]
    result_dict_sport_4gram[key] = value

#print(result_dict_sport_4gram)

# COMMAND ----------

wc4_cell = WordCloud(background_color='black', max_words=100, min_font_size=8, colormap='rainbow_r')
wc4_sport = WordCloud(background_color='white', max_words=100, min_font_size=8, colormap='rainbow_r')

fig = plt.figure(figsize=(20, 20))
wc4_cell = wc4_cell.generate_from_frequencies(result_dict_cell_4gram)
wc4_sport = wc4_sport.generate_from_frequencies(result_dict_sport_4gram)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 4-gram')
plt.imshow(wc4_cell)
plt.axis('off')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('sport 4-gram')
plt.imshow(wc4_sport)
plt.axis('off')
plt.show()

# COMMAND ----------

wc4_cell = WordCloud(background_color='black', max_words=100, min_font_size=8, colormap='rainbow_r')
wc4_sport = WordCloud(background_color='white', max_words=100, min_font_size=8, colormap='rainbow_r')

fig = plt.figure(figsize=(20, 20))
wc4_cell = wc4_cell.generate_from_frequencies(result_dict_cell_4gram)
wc4_sport = wc4_sport.generate_from_frequencies(result_dict_sport_4gram)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('cell 4-gram')
plt.imshow(wc4_cell)
plt.axis('off')

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('sport 4-gram')
plt.imshow(wc4_sport)
plt.axis('off')
plt.show()

# COMMAND ----------


