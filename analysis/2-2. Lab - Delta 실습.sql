-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md <i18n value="a209ac48-08a6-4b89-b728-084a515fd335"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC # Delta Lake 테입블 조작하기
-- MAGIC
-- MAGIC 이 노트북은 Delta Lake의 기본 기능 중 일부에 대한 실습을 제공합니다.
-- MAGIC
-- MAGIC ## 학습 목표
-- MAGIC 이 실습을 마치면 다음을 수행할 수 있습니다.
-- MAGIC - 다음을 포함하여 Delta Lake 테이블을 만들고 조작하는 표준 작업을 실행합니다.
-- MAGIC   - **`CREATE TABLE`**
-- MAGIC   - **`INSERT INTO`**
-- MAGIC   - **`SELECT FROM`**
-- MAGIC   - **`UPDATE`**
-- MAGIC   - **`DELETE`**
-- MAGIC   - **`MERGE`**
-- MAGIC   - **`DROP TABLE`**

-- COMMAND ----------

-- MAGIC %md <i18n value="6582dbcd-72c7-496b-adbd-23aef98e20e9"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## 셋업
-- MAGIC 다음 스크립트를 실행하여 필요한 변수를 설정하고 이 노트북의 과거 실행을 지웁니다. 이 셀을 다시 실행하면 실습을 다시 시작할 수 있습니다.

-- COMMAND ----------

-- MAGIC %run ../Includes/Setup

-- COMMAND ----------

-- MAGIC %md <i18n value="0607f2ed-cfe6-4a38-baa4-e6754ec1c664"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## Table 생성
-- MAGIC
-- MAGIC 이 노트북에서 bean 컬렉션을 추적하는 테이블을 만들 것입니다.
-- MAGIC
-- MAGIC 아래 셀을 사용하여 **`beans`** 라는 관리형 Delta Lake 테이블을 만듭니다.
-- MAGIC
-- MAGIC 아래 스키마가 주어집니다:
-- MAGIC
-- MAGIC | 필드 이름 | 필드 타입 |
-- MAGIC | --- | --- |
-- MAGIC | name | STRING |
-- MAGIC | color | STRING |
-- MAGIC | grams | FLOAT |
-- MAGIC | delicious | BOOLEAN |

-- COMMAND ----------

-- TODO
<FILL-IN>


-- COMMAND ----------

-- MAGIC %md <i18n value="2167d7d7-93d1-4704-a7cb-a0335eaf8da7"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC **NOTE**: 실습 전체에서 Python을 사용하여 사용자의 수행에 대한 검사를 실행합니다.다음 셀은 지침을 따르지 않은 경우 변경해야 할 사항에 대한 메시지와 함께 오류로 반환됩니다. 셀 실행 결과가 출력되지 않으면 이 단계가 완료되었음을 의미합니다.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assert spark.table("beans"), "`beans` 로 명명된 테이블이 없습니다"
-- MAGIC assert spark.table("beans").columns == ["name", "color", "grams", "delicious"], "컬럼의 이름 및 순서가 위에서 명시된 순서인지 확인하세요"
-- MAGIC assert spark.table("beans").dtypes == [("name", "string"), ("color", "string"), ("grams", "float"), ("delicious", "boolean")], "컬럼 유형이 위에 제공된 것과 동일한지 확인하십시오."

-- COMMAND ----------

-- MAGIC %md <i18n value="89004ef0-db16-474b-8cce-eff85c225a65"/>
-- MAGIC
-- MAGIC
-- MAGIC ## 데이터 삽입
-- MAGIC
-- MAGIC 다음 셀을 실행하여 테이블에 세 개의 행을 삽입합니다.

-- COMMAND ----------

create table asac.beans_jyp
  (name STRING, color STRING, gram FLOAT, delicious BOOLEAN);

-- COMMAND ----------

SELECT * FROM asac.beans_jyp

-- COMMAND ----------

INSERT INTO asac.beans_jyp VALUES
("black", "black", 500, true),
("lentils", "brown", 1000, true),
("jelly", "rainbow", 42.5, false)

-- COMMAND ----------

-- MAGIC %md <i18n value="48d649a1-cde1-491f-a90d-95d2e336e140"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 테이블 내용을 수동으로 검토하여 데이터가 예상대로 작성되었는지 확인하십시오.

-- COMMAND ----------

-- TODO
 <FILL-IN>

-- COMMAND ----------

-- MAGIC %md <i18n value="f0406eef-6973-47c9-8f89-a667c53cfea7"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 아래 제공된 추가 레코드를 삽입하십시오. 이것을 단일 트랜잭션으로 실행해야 합니다.

-- COMMAND ----------

-- TODO
-- <FILL-IN>
insert into asac.beans_jyp values 
('pinto', 'brown', 1.5, true),
('green', 'green', 178.3, true),
('beanbag chair', 'white', 40000, false)

-- COMMAND ----------

-- MAGIC %md <i18n value="e1764f9f-8052-47bb-a862-b52ca438378a"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 아래 셀을 실행하여 데이터가 적절한 상태인지 확인하십시오.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assert spark.table("asac.beans_jyp").count() == 6, "테이블에는 6개의 레코드가 있어야 합니다."
-- MAGIC assert spark.conf.get("spark.databricks.delta.lastCommitVersionInSession") == "2", "테이블에 커밋이 3개만 있어야 합니다."
-- MAGIC assert set(row["name"] for row in spark.table("asac.beans_jyp").select("name").collect()) == {'beanbag chair', 'black', 'green', 'jelly', 'lentils', 'pinto'}, "제공된 데이터를 수정하지 않았는지 확인하십시오"

-- COMMAND ----------

-- MAGIC %md <i18n value="e38adafa-bd10-4191-9c69-e6a4363532ec"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## 레코드 업데이트
-- MAGIC
-- MAGIC 친구가 bean 재고를 검토하고 있습니다. 많은 토론 끝에 당신은 jelly bean 이 맛있다는 데 동의합니다.
-- MAGIC
-- MAGIC 다음 셀을 실행하여 이 레코드를 업데이트하십시오.

-- COMMAND ----------

UPDATE asac.beans_jyp
SET delicious = true
WHERE name = "jelly"

-- COMMAND ----------

-- MAGIC %md <i18n value="d8637fab-6d23-4458-bc08-ff777021e30c"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 실수로 `pinto` bean의 무게를 잘못 입력했음을 알게 됩니다.
-- MAGIC
-- MAGIC 이 레코드의 **`grams`** 열을 올바른 무게인 1500으로 업데이트합니다.

-- COMMAND ----------

-- TODO
<FILL-IN>

-- COMMAND ----------

-- MAGIC %md <i18n value="954bf892-4db6-4b25-9a9a-83b0f6ecc123"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 아래 셀을 실행하여 제대로 완료되었는지 확인합니다.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assert spark.table("asac.beans_jyp").filter("name='pinto'").count() == 1, "pinto bean 에 대한 항목은 1개만 존재 해야 합니다."
-- MAGIC row = spark.table("asac.beans_jyp").filter("name='pinto'").first()
-- MAGIC assert row["color"] == "brown", "pinto bean은 brown 으로 표시되어야 합니다."
-- MAGIC assert row["grams"] == 1500, "`grams` 을 1500 으로 입력했는지 확인하세요"
-- MAGIC assert row["delicious"] == True, "pinto bean 은 delicious bean 입니다"

-- COMMAND ----------

-- MAGIC %md <i18n value="f36d551a-f588-43e4-84a2-6aa49a420c04"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## 레코드 삭제
-- MAGIC
-- MAGIC 맛있는 bean만 추적하기로 결정했습니다.
-- MAGIC
-- MAGIC 맛있지 않은 모든 bean을 삭제하는 쿼리를 실행합니다.

-- COMMAND ----------

-- TODO
<FILL-IN>

-- COMMAND ----------

-- MAGIC %md <i18n value="1c8d924c-3e97-49a0-b5e4-0378c5acd3c8"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 다음 셀을 실행하여 이 작업이 성공했는지 확인합니다.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assert spark.table("beans").filter("delicious=true").count() == 5, "5개의 delicious bean 이 테이블에 존재해야 합니다."
-- MAGIC assert spark.table("beans").filter("name='beanbag chair'").count() == 0, "로직이 non-delicious beans 을 삭제 했는지 확인하세요."

-- COMMAND ----------

-- MAGIC %md <i18n value="903473f1-ddca-41ea-ae2f-dc2fac64936e"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## 레코드를 Upsert 하기 위해 Merge문 사용
-- MAGIC
-- MAGIC 당신의 친구로 부터 새로운 bean을 받았습니다. 아래 셀은 이를 temp view로 등록합니다.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW new_beans(name, color, grams, delicious) AS VALUES
('black', 'black', 60.5, true),
('lentils', 'green', 500, true),
('kidney', 'red', 387.2, true),
('castor', 'brown', 25, false);

SELECT * FROM new_beans

-- COMMAND ----------

-- MAGIC %md <i18n value="58d50e50-65f1-403b-b74e-1143cde49356"/>
-- MAGIC
-- MAGIC 아래 셀에서 위에서 생성한 view를 사용하여 **`beans`** 테이블에 새 레코드를 하나의 트랜잭션으로 업데이트하고 삽입하는 병합 문을 작성합니다.
-- MAGIC
-- MAGIC 아래 로직을 확인하십시오.
-- MAGIC - name **AND** color으로 bean을 일치시킵니다.
-- MAGIC - 기존 grams에 새 grams를 추가하여 기존 bean을 업데이트합니다.
-- MAGIC - 새 bean은 delicious가 true 일 때만 삽입

-- COMMAND ----------

-- TODO
<FILL-IN>

-- COMMAND ----------

-- MAGIC %md <i18n value="9fbb65eb-9119-482f-ab77-35e11af5fb24"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 아래 셀을 실행하여 작업을 확인하십시오.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pyspark.sql.functions as F
-- MAGIC last_version = spark.sql("DESCRIBE HISTORY beans").orderBy(F.col("version").desc()).first()
-- MAGIC
-- MAGIC assert last_version["operation"] == "MERGE", "트랜잭션은 MERGE로 수행되어야 합니다."
-- MAGIC
-- MAGIC metrics = last_version["operationMetrics"]
-- MAGIC assert metrics["numOutputRows"] == "5", "delicious bean 만 삽입했는지 확인하세요."
-- MAGIC assert metrics["numTargetRowsUpdated"] == "1", "name 과 color 필드로 데이터를 match 했는지 확인하세요."
-- MAGIC assert metrics["numTargetRowsInserted"] == "2", "새로 수집된 bean만 삽입했는지 확인하세요."
-- MAGIC assert metrics["numTargetRowsDeleted"] == "0", "어떤 rows 도 삭제되어서는 안됩니다."

-- COMMAND ----------

-- MAGIC %md <i18n value="4a668d7c-e16b-4061-a5b7-1ec732236308"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## 테이블 DROP
-- MAGIC
-- MAGIC 관리형 Delta Lake 테이블로 작업할 때 테이블을 삭제하면 테이블 및 모든 기본 데이터 파일에 대한 액세스 권한이 영구적으로 삭제됩니다.
-- MAGIC
-- MAGIC **참고**: 이 과정의 뒷부분에서 Delta Lake 테이블을 파일 모음으로 접근하고 다른 영구 지속성을 보장하는 external table에 대해 배우게 됩니다.
-- MAGIC
-- MAGIC 아래 셀에 **`beans`** 테이블을 삭제하는 쿼리를 작성하세요.

-- COMMAND ----------

-- TODO
<FILL-IN>

-- COMMAND ----------

-- MAGIC %md <i18n value="4cc5c126-5e56-423e-a814-f6c422312802"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC 아래 셀을 실행하여 테이블이 더 이상 존재하지 않는지 확인 하십시오

-- COMMAND ----------

-- MAGIC %python
-- MAGIC assert spark.sql("SHOW TABLES LIKE 'beans'").collect() == [], "현재 데이터베이스에서 `beans` 테이블을 삭제했는지 확인하십시오."

-- COMMAND ----------

-- MAGIC %md <i18n value="f4d330e3-dc40-4b6e-9911-34902bab22ae"/>
-- MAGIC
-- MAGIC
-- MAGIC
-- MAGIC ## Wrapping Up
-- MAGIC
-- MAGIC 이 실습을 완료하면 이제 아래 항목을 수행할 수 있습니다.
-- MAGIC * 표준 Delta Lake 테이블 생성 및 데이터 조작 명령 완료

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

-- COMMAND ----------


