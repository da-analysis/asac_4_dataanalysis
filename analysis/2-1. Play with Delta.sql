-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Databricks Delta Quickstart
-- MAGIC 이 노트북에서는 SparkSQL 을 사용해서 아래와 같이 Delta Lake 형식의 데이터를 다루는 다양한 방법에 대해서 다룹니다.  
-- MAGIC
-- MAGIC
-- MAGIC * Delta Table을 만들고 다양한 DML문들을 사용해서 데이터를 수정하고 정제
-- MAGIC * Delta Table의 구조 이해
-- MAGIC * Time Travel 을 이용한 Table History 관리   
-- MAGIC * 댜양한 IO 최적화 기능 

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC ## Run Setup
-- MAGIC 가장 먼저 할 일은 설정 스크립트를 실행하는 것입니다. 각 사용자로 범위가 지정된 사용자 이름 및 데이터베이스를 정의합니다.

-- COMMAND ----------

-- DBTITLE 1,setup
-- MAGIC %run ../Includes/Setup

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ##![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Creating a Delta Table
-- MAGIC
-- MAGIC Delta Lake로 테이블을 생성하기 위해 작성해야 하는 코드가 많지 않습니다. 과정 전반에 걸쳐 보게 될 Delta Lake 테이블을 만드는 방법에는 여러 가지가 있습니다. 가장 쉬운 방법 중 하나인 Delta Lake 테이블 등록부터 시작하겠습니다.
-- MAGIC
-- MAGIC
-- MAGIC 필요한 것: 
-- MAGIC - **`CREATE TABLE`** 문장
-- MAGIC - 테이블 이름 (아래 예제에서 **`students_JYP`** 사용)
-- MAGIC - 스키마
-- MAGIC
-- MAGIC **NOTE:** Databricks Runtime 8.0 이상에서는 Delta Lake가 기본 형식이며 **`USING DELTA`** 가 필요하지 않습니다.

-- COMMAND ----------

create schema asac_jyp

-- COMMAND ----------

drop table asac_jyp.students_JYP 

-- COMMAND ----------

drop schema asac_jyp

-- COMMAND ----------



-- COMMAND ----------

-- DBTITLE 1,Delta Table의 생성
CREATE TABLE asac.students_JYP 
  (id INT, name STRING, value DOUBLE);

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC 돌아가서 위의셀을 다시 실행하려고 하면 오류가 발생합니다! 이것은 예상된 것입니다. 테이블이 이미 존재하기 때문에 오류가 발생합니다.
-- MAGIC
-- MAGIC 테이블이 존재하는지 확인하는 추가 구문 **`IF NOT EXISTS`** 를 추가할 수 있습니다. 이러면 테이블이 이미 존재할 경우 오류를 회피할 것 입니다.

-- COMMAND ----------

-- DBTITLE 1,IF NOT EXISTS 추가해서 생성하기
CREATE TABLE IF NOT EXISTS students_JYP 
  (id INT, name STRING, value DOUBLE)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Full DML Support
-- MAGIC <br>
-- MAGIC 일반적인 Data Lake의 경우 모든 데이터는 Append Only임을 가정합니다.  
-- MAGIC
-- MAGIC Delta Lake를 사용하면 마치 Database를 사용하는 것처럼 Insert,Update,Delete를 사용해서 손쉽게 데이터셋을 수정할 수 있습니다. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ### 데이터 삽입
-- MAGIC 대부분의 경우 데이터는 다른 소스의 쿼리 결과로 테이블에 삽입됩니다.
-- MAGIC
-- MAGIC 그러나 표준 SQL에서와 마찬가지로 여기에 표시된 대로 값을 직접 삽입할 수도 있습니다.

-- COMMAND ----------

INSERT INTO students_JYP VALUES (1, "Yve", 1.0);
INSERT INTO students_JYP VALUES (2, "Omar", 2.5);
INSERT INTO students_JYP VALUES (3, "Elia", 3.3);

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 위의 셀에서 세 개의 별도 **`INSERT`** 문을 완료했습니다. 이들 각각은 자체 ACID 보장과 함께 별도의 트랜잭션으로 처리됩니다. 
-- MAGIC  
-- MAGIC 대부분의 경우 아래와 같이 단일 트랜잭션에 많은 레코드를 삽입합니다.

-- COMMAND ----------

INSERT INTO students_JYP
VALUES 
  (4, "Ted", 4.7),
  (5, "Tiffany", 5.5),
  (6, "Vini", 6.3);
  

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC Databricks에는 **`COMMIT`** 키워드가 없습니다. 트랜잭션은 실행되는 즉시 실행되고 성공하면 커밋됩니다.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC ### Delta Table 조회하기
-- MAGIC
-- MAGIC Delta Lake 테이블을 조회하는 것이 표준 **`SELECT`** 문을 사용하는 것만큼 쉽습니다.

-- COMMAND ----------

SELECT * FROM students_JYP

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC ### 레코드 업데이트
-- MAGIC
-- MAGIC 레코드 업데이트는 원자성(Atomic) 보장도 제공합니다. 테이블의 현재 버전에 대한 스냅샷 읽기를 수행하고 **`WHERE`** 절과 일치하는 모든 필드를 찾은 다음 설명된 대로 변경 사항을 적용합니다.
-- MAGIC
-- MAGIC 아래에서 이름이 문자 **T**로 시작하는 모든 학생을 찾고 **`value`** 열의 숫자에 1을 더합니다.

-- COMMAND ----------

UPDATE students_JYP 
SET value = value + 1
WHERE name LIKE "T%"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 변경 사항이 적용되었는지 확인하기위해서 테이블을 다시 쿼리하십시오.

-- COMMAND ----------

SELECT * FROM students_JYP

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ### 레코드 삭제
-- MAGIC
-- MAGIC 삭제도 원자성을 가지므로 데이터 레이크하우스에서 데이터를 제거할 때 부분적으로만 성공할 위험이 없습니다.
-- MAGIC
-- MAGIC **`DELETE`** 문은 하나 이상의 레코드를 제거할 수 있지만 항상 단일 트랜잭션이 발생합니다.

-- COMMAND ----------

DELETE FROM students_JYP 
WHERE value > 6

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) MERGE 를 사용한 Upsert 수행 
-- MAGIC
-- MAGIC 일부 SQL 시스템에는 업데이트, 삽입 및 기타 데이터 조작을 단일 명령으로 실행할 수 있는 upsert 개념이 있습니다.
-- MAGIC
-- MAGIC Databricks는 **`MERGE`** 키워드를 사용하여 이 작업을 수행합니다.
-- MAGIC
-- MAGIC CDC(변경 데이터 캡처) 피드에서 출력할 수 있는 4개의 레코드가 포함된 다음 Temp View 가 있다고 가정해 봅시다

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW updates_JYP(id, name, value, type) AS VALUES
  (2, "Omar", 15.2, "update"),
  (3, "", null, "delete"),
  (7, "Blue", 7.7, "insert"),
  (11, "Diya", 8.8, "update");
  
SELECT * FROM updates_JYP;

-- COMMAND ----------

-- MAGIC %md <i18n value="6fe009d5-513f-4b93-994f-1ae9a0f30a80"/>
-- MAGIC
-- MAGIC 지금까지 본 구문을 사용하여 유형별로 이 보기에서 필터링하여 각각 레코드를 삽입, 업데이트 및 삭제하는 3개의 문을 작성할 수 있습니다.  
-- MAGIC 그러나 이렇게 하면 3개의 개별 트랜잭션이 발생합니다. 이러한 트랜잭션 중 하나라도 실패하면 데이터가 유효하지 않은 상태로 남을 수 있습니다.
-- MAGIC
-- MAGIC 대신 이러한 작업을 단일 원자 트랜잭션으로 결합하여 3가지 유형의 변경 사항을 함께 적용합니다.
-- MAGIC
-- MAGIC **`MERGE`** 문에는 일치(ON 절)시킬 필드가 하나 이상 있어야 하며 각 **`WHEN MATCHED`** 또는 **`WHEN NOT MATCHED`** 절에는 원하는 만큼 추가 조건문이 있을 수 있습니다.
-- MAGIC
-- MAGIC 여기에서 **`id`** 필드를 일치시킨 다음 **`type`** 필드를 필터링하여 레코드를 적절하게 업데이트, 삭제 또는 삽입합니다.

-- COMMAND ----------

MERGE INTO students_JYP b
USING updates_JYP u
ON b.id=u.id
WHEN MATCHED AND u.type = "update"
  THEN UPDATE SET *
WHEN MATCHED AND u.type = "delete"
  THEN DELETE
WHEN NOT MATCHED AND u.type = "insert"
  THEN INSERT *

-- COMMAND ----------

SELECT * FROM students_JYP;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC 3개의 레코드만 **`MERGE`** 문의 영향을 받았습니다. updates_JYP 테이블의 레코드 중 하나는 학생 테이블에 일치하는 **`id`** 가 없지만 **`update`** 로 표시되었고 사용자가 지정한 규칙에 따라 이 레코드를 삽입하는 대신 무시했습니다.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Transaction Log 
-- MAGIC Delta Lake를 이루는 테이블의 내부 구조를 알아보도록 하겠습니다.

-- COMMAND ----------

-- DBTITLE 1,아래 명령을 통해 Location 행에서 테이블을 이루는 파일의 위치 정보를 확인하기
DESCRIBE EXTENDED students_JYP

-- COMMAND ----------

-- DBTITLE 1,Delta Lake File 을 조사해 보기.
-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/students_jyp"))

-- COMMAND ----------

-- DBTITLE 1,Delta 의 트랙젝션 로그를 살펴보기
-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/students_jyp/_delta_log"))

-- COMMAND ----------

-- DBTITLE 1,트랙젝션 로그 중 하나를 열어 보기
-- MAGIC %python
-- MAGIC display(spark.sql(f"SELECT * FROM json.`dbfs:/user/hive/warehouse/students_jyp/_delta_log/00000000000000000006.json`"))

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Time Travel 기능 

-- COMMAND ----------

DESCRIBE HISTORY students_JYP

-- COMMAND ----------

-- DBTITLE 1,과거 버전의 데이터 조회
SELECT * FROM students_JYP VERSION AS OF 2;
-- SELECT * FROM students_JYP@v2;

-- COMMAND ----------

-- DBTITLE 1,과거 버전으로 돌아가기
RESTORE TABLE students_JYP TO VERSION AS OF 2;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Data file에 대한 OPTIMIZE 
-- MAGIC
-- MAGIC 이런저런 작업을 하다 보면 필연적으로 굉장히 작은 데이터 파일들이 많이 생성되게 됩니다.  
-- MAGIC 성능 향상을 위해서 이런 파일들에 대한 최적화하는 방법과 불필요한 파일들을 정리하는 명령어들에 대해서 알아봅시다. 

-- COMMAND ----------

DESCRIBE DETAIL students_JYP

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **`OPTIMIZE`** 명령어는 기존의 데이터 파일내의 레코드들을 합쳐서 새로 최적의 사이즈로 파일을 만들고 기존의 작은 파일들을 읽기 성능이 좋은 큰 파일들로 대체합니다.  
-- MAGIC 이 떄 옵션값으로 하나 이상의 필드를 지정해서 **`ZORDER`** 인덱싱을 수행할 수 있습니다.  
-- MAGIC Z-Ordering은 관련 정보를 동일한 파일 집합에 배치해서 읽어야 하는 데이터의 양을 줄여 쿼리 성능을 향상 시키는 기술입니다. 쿼리 조건에 자주 사용되고 해당 열에 높은 카디널리티(distinct 값이 많은)가 있는 경우 `ZORDER BY`를 사용합니다.

-- COMMAND ----------

OPTIMIZE students_JYP ZORDER BY id

-- COMMAND ----------

-- DBTITLE 1,DESCRIBE HISTORY 를 통해서 테이블의 이력을 확인하기
DESCRIBE HISTORY students_JYP

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Stale File 정리하기
-- MAGIC Databricks는 자동으로 Delta Lake Table 에서 불필요한 파일들을 정리합니다.  
-- MAGIC Delta Lake의 Versioning과 Time Travel은 과거 버전을 조회하고 실수했을 경우 데이터를 rollback하는 매우 유용한 기능이지만, 데이터 파일의 모든 버전을 영구적으로 저장하는 것은 비용이 많이 들게 됩니다.  
-- MAGIC 기본값으로 **VACUUM** 은 7일 미만의 데이터를 삭제하지 못하도록 합니다. 아래의 예제는 이 기본 설정을 무시하고 가장 최근 버전 데이터만 남기고 모든 과거 버전의 stale file을 정리하는 예제입니다. 

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/students_jyp"))

-- COMMAND ----------

SET spark.databricks.delta.retentionDurationCheck.enabled = false;
SET spark.databricks.delta.vacuum.logging.enabled = true;

VACUUM students_JYP RETAIN 0 HOURS

-- COMMAND ----------

-- DBTITLE 1,현재 상태의 데이터 파일들만 남아있는지 확인하기
-- MAGIC %python
-- MAGIC table_path="/user/hive/warehouse/students_jyp"
-- MAGIC #print(table_path)
-- MAGIC display(dbutils.fs.ls(table_path))

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Table Drop
-- MAGIC
-- MAGIC 대상 테이블에 대한 적절한 권한이 있다고 가정하면 **`DROP TABLE`** 명령을 사용하여 레이크하우스의 데이터를 영구적으로 삭제할 수 있습니다.
-- MAGIC
-- MAGIC **NOTE**: 이 과정의 뒷부분에서 테이블 ACL(액세스 제어 목록) 및 기본 권한에 대해 설명합니다. 적절하게 구성된 레이크하우스에서 사용자는 프로덕션 테이블을 삭제할 수 **없어야** 합니다.

-- COMMAND ----------

DROP TABLE students_JYP

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

-- COMMAND ----------


