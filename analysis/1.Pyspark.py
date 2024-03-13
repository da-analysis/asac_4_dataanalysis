# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC # Pyspark
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC
# MAGIC ## 참고 문서
# MAGIC
# MAGIC [Recommenders Microsoft Github](https://github.com/microsoft/recommenders)
# MAGIC
# MAGIC [Python 데이터 분석 실무](https://wikidocs.net/16565)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect SparkContext
# MAGIC
# MAGIC ![SparkContext](https://www.tutorialspoint.com/pyspark/images/sparkcontext.jpg)

# COMMAND ----------

sc.version

# COMMAND ----------

sc.pythonVer

# COMMAND ----------

str(sc.sparkHome)

# COMMAND ----------

sc.appName

# COMMAND ----------

sc.applicationId

# COMMAND ----------

sc.getConf().getAll()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Tutorial

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS diamonds;
# MAGIC
# MAGIC CREATE TABLE diamonds
# MAGIC USING csv
# MAGIC OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diamonds

# COMMAND ----------

diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")
diamonds.write.mode('overwrite').format("delta").save("/delta/diamonds")

diamonds.createOrReplaceTempView("diamonds")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS diamonds;
# MAGIC
# MAGIC CREATE TABLE diamonds USING DELTA LOCATION '/delta/diamonds/'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diamonds

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### SparkR
# MAGIC
# MAGIC - Pyspark, SparkR 은 Spark Dataframe View 에서 같은 데이터를 조회할 수 있음
# MAGIC - 또는 SparkR 에서 직접 데이터 로드해도 됨

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC
# MAGIC test_df <- sql("select * from diamonds")
# MAGIC
# MAGIC head(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RDD (기초)
# MAGIC
# MAGIC ![Cheat Sheet](https://storage.ning.com/topology/rest/1.0/file/get/2808331195?profile=original)

# COMMAND ----------

rdd = sc.parallelize([('a', 7), ('a', 2), ('b', 2)])
rdd.take(2)

# COMMAND ----------

rdd.first()

# COMMAND ----------

rdd.top(2)

# COMMAND ----------

rdd.min()

# COMMAND ----------

rdd.count()

# COMMAND ----------

rdd2 = sc.parallelize(range(100))

rdd2.take(10)

# COMMAND ----------

rdd.countByKey()

# COMMAND ----------

rdd.countByValue()

# COMMAND ----------

rdd.collectAsMap()

# COMMAND ----------

rdd.map(lambda x: x+(x[1], x[0])).collect()

# COMMAND ----------

rdd.reduceByKey(lambda x,y : x+y).collect()

# COMMAND ----------

rdd.saveAsTextFile("/FileStore/tables/syleeie/rdd/test")

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/tables/syleeie/rdd/test"))

# COMMAND ----------

rdd.groupByKey() \
   .mapValues(list) \
   .collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data.Frame
# MAGIC ### read.csv

# COMMAND ----------

diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")
diamonds.registerTempTable("diamonds")

# COMMAND ----------

diamonds.head()

# COMMAND ----------

diamonds.show()

# COMMAND ----------

diamonds.dtypes

# COMMAND ----------

diamonds.take(3)

# COMMAND ----------

diamonds.printSchema()

# COMMAND ----------

diamonds.explain()

# COMMAND ----------

from pyspark.sql import functions as F
diamonds.select("carat").show()

# COMMAND ----------

diamonds.select("carat", "color").show()

# COMMAND ----------

 diamonds.select(diamonds['carat'] > 0.3).show()

# COMMAND ----------

 diamonds.select('cut', diamonds.cut.substr(1,3).alias('test_nm')).show()

# COMMAND ----------

 diamonds.select('carat', diamonds.carat.between(0.2, 0.25)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Column (add, update, remove)

# COMMAND ----------

diamonds = diamonds.withColumn('zero_value', F.lit(0))
diamonds.show()

# COMMAND ----------

diamonds = diamonds.withColumnRenamed('zero_value', 'zero_value2')
diamonds.show()

# COMMAND ----------

diamonds = diamonds.drop('zero_value2')
diamonds.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## GroupBy

# COMMAND ----------

diamonds.groupBy("color") \
        .count() \
        .show()

# COMMAND ----------

diamonds.groupBy("color") \
        .agg(F.min("price"), F.max("price"), F.avg("price")) \
        .show()             

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter

# COMMAND ----------

diamonds.count()

# COMMAND ----------

diamonds.filter(diamonds['carat'] <= 0.3).count()

# COMMAND ----------

diamonds.filter(diamonds['carat'] <= 0.3).show()

# COMMAND ----------

 diamonds.sort("carat", ascending=False).show()

# COMMAND ----------

diamonds.describe().show()

# COMMAND ----------

from pyspark.sql.functions import avg

display(diamonds.select("color","price").groupBy("color").agg(avg("price")).sort("color"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark.SQL
# MAGIC
# MAGIC [Spark SQL 가이드](https://spark.apache.org/docs/2.3.0/api/sql/index.html)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diamonds

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT color, avg(price) AS price FROM diamonds GROUP BY color ORDER BY color

# COMMAND ----------

# MAGIC %md
# MAGIC ### SQL 짧게 배우기

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS customer;
# MAGIC
# MAGIC create table customer (
# MAGIC     userid    integer,
# MAGIC     username  varchar(10),
# MAGIC     join_date varchar(10)
# MAGIC );
# MAGIC
# MAGIC insert into customer
# MAGIC select 1 as userid, 'A' as username, '2015-08-01' as join_date union all
# MAGIC select 2 as userid, 'B' as username, '2015-08-02' as join_date union all
# MAGIC select 3 as userid, 'C' as username, '2015-08-01' as join_date union all
# MAGIC select 4 as userid, 'D' as username, '2015-08-03' as join_date union all
# MAGIC select 5 as userid, 'E' as username, '2015-08-07' as join_date union all
# MAGIC select 6 as userid, 'F' as username, '2015-08-22' as join_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from customer

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS order_table;
# MAGIC
# MAGIC create table order_table (
# MAGIC     userid     integer,
# MAGIC     order_date varchar(10),
# MAGIC     method     varchar(10),
# MAGIC     amount     integer,
# MAGIC     discount   numeric
# MAGIC );
# MAGIC
# MAGIC insert into order_table 
# MAGIC select 1 as userid, '2015-08-03' as method, 'TOUCH' as method, 10000 as amount, null as discount union all
# MAGIC select 1 as userid, '2015-08-10' as method, 'TOUCH' as method, 10000 as amount, -950.4 as discount union all
# MAGIC select 1 as userid, '2015-08-14' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 1 as userid, '2015-08-12' as method, 'TOUCH' as method, 10000 as amount, null as discount union all
# MAGIC select 2 as userid, '2015-08-03' as method, 'TOUCH' as method, 5000 as amount, -500 as discount union all
# MAGIC select 2 as userid, '2015-08-11' as method, 'TOUCH' as method, 5000 as amount, -300 as discount union all
# MAGIC select 2 as userid, '2015-08-12' as method, 'TOUCH' as method, 5000 as amount, -700 as discount union all
# MAGIC select 2 as userid, '2015-08-22' as method, 'TOUCH' as method, 5000 as amount, -1000 as discount union all
# MAGIC select 2 as userid, '2015-08-28' as method, 'TOUCH' as method, 5000 as amount, -600 as discount union all
# MAGIC select 3 as userid, '2015-08-07' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 3 as userid, '2015-08-19' as method, 'TOUCH' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 3 as userid, '2015-08-30' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 4 as userid, '2015-08-05' as method, 'CALL' as method, 20000 as amount, -3000 as discount union all
# MAGIC select 4 as userid, '2015-08-18' as method, 'TOUCH' as method, 30000 as amount, -5000 as discount union all
# MAGIC select 5 as userid, '2015-08-05' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 5 as userid, '2015-08-17' as method, 'CALL' as method, 10000 as amount, null as discount union all
# MAGIC select 5 as userid, '2015-08-21' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 5 as userid, '2015-08-23' as method, 'CALL' as method, 10000 as amount, -1000 as discount union all
# MAGIC select 5 as userid, '2015-08-29' as method, 'CALL' as method, 10000 as amount, -1000 as discount

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct order_date from order_table order by 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table limit 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table where order_date =  '2015-08-05';

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table where order_date between '2015-08-05' and '2015-08-20' order by order_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table where order_date in ('2015-08-05', '2015-08-07');

# COMMAND ----------

# MAGIC %sql
# MAGIC select userid
# MAGIC      , orders
# MAGIC      , gmv
# MAGIC      , case orders
# MAGIC            when 1 then 'A'
# MAGIC            when 2 then 'B'
# MAGIC            when 3 then 'C'
# MAGIC            else        'D'
# MAGIC        end as simple_case
# MAGIC      , case when orders >= 4 and gmv >= 40000 then 'A'
# MAGIC             when orders >= 3 and gmv >= 30000 then 'B'
# MAGIC             when orders >= 2 and gmv >= 10000 then 'C'
# MAGIC             else                                   'D'
# MAGIC        end as searched_case
# MAGIC from (select userid
# MAGIC              , count(userid) as orders
# MAGIC              , sum(amount)   as gmv
# MAGIC       from order_table
# MAGIC group by userid) ord;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_table where discount is null;

# COMMAND ----------

# MAGIC %sql
# MAGIC select userid
# MAGIC      , min(order_date) as first_order_date
# MAGIC      , max(order_date) as last_order_date
# MAGIC      , sum(amount)     as amount
# MAGIC from order_table
# MAGIC group by userid
# MAGIC having sum(amount) >= 30000
# MAGIC order by amount desc;

# COMMAND ----------

# MAGIC %sql
# MAGIC select cus.userid
# MAGIC      , count(ord.userid) as orders
# MAGIC from customer as cus
# MAGIC inner join order_table as ord
# MAGIC on cus.userid = ord.userid
# MAGIC group by cus.userid
# MAGIC order by cus.userid;

# COMMAND ----------

# MAGIC %sql
# MAGIC select cus.userid
# MAGIC      , count(ord.userid) as orders
# MAGIC from customer as cus
# MAGIC left outer join order_table as ord
# MAGIC on cus.userid = ord.userid
# MAGIC group by cus.userid
# MAGIC order by cus.userid

# COMMAND ----------

# MAGIC %sql
# MAGIC select cus.userid
# MAGIC      , count(ord.userid) as orders
# MAGIC from customer as cus
# MAGIC cross join order_table as ord
# MAGIC group by cus.userid
# MAGIC order by cus.userid

# COMMAND ----------

# MAGIC %sql
# MAGIC select ord.*
# MAGIC from (select ord.userid
# MAGIC              , ord.order_date
# MAGIC              , ord.amount
# MAGIC              , row_number() over (partition by userid order by order_date) as rk
# MAGIC from order_table as ord) ord
# MAGIC where ord.rk = 1
