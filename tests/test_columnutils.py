import pytest
from tidypyspark.column_utils import case_when, ifelse

def test_case_when():
    
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  from pyspark.sql import Row

  df = spark.createDataFrame([("a", 1), ("b", 2), ("c", 3)],
                              ["letter", "number"]
                            )
  # +------+------+
  # |letter|number|
  # +------+------+
  # |     a|     1|
  # |     b|     2|
  # |     c|     3|
  # +------+------+

  # Test with case_when
  df1 = df.withColumn("new_number",
                case_when([(F.col("number") == 1, F.lit(10)),
                            (F.col("number") == 1, F.lit(20)),
                            (F.col("number") == 3, F.lit(30))],
                            default = F.lit(0)
                          )
                )
  # Output
  # +------+------+----------+
  # |letter|number|new_number|
  # +------+------+----------+
  # |     a|     1|        10|
  # |     b|     2|         0|
  # |     c|     3|        30|
  # +------+------+----------+
  assert df1.collect() == ([Row(letter='a', number=1, new_number=10), 
                           Row(letter='b', number=2, new_number=0), 
                           Row(letter='c', number=3, new_number=30)]
                          )

  # Test with case_when and no default
  df2 = df.withColumn("new_number",
                case_when([(F.col("number") == 1, F.lit(10)),
                            (F.col("number") == 2, F.lit(20)),
                            (F.col("number") == 4, F.lit(40))])
                )
  # Output
  # +------+------+----------+
  # |letter|number|new_number|
  # +------+------+----------+
  # |     a|     1|        10|
  # |     b|     2|        20|
  # |     c|     3|      null|
  # +------+------+----------+
  assert df2.collect() == ([Row(letter='a', number=1, new_number=10),
                            Row(letter='b', number=2, new_number=20),
                            Row(letter='c', number=3, new_number=None)]
                          )

def test_if_else():
  
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  from pyspark.sql import Row

  df = spark.createDataFrame([("a", 1), ("b", 2), ("c", 3)],
                              ["letter", "number"]
                            )
  # +------+------+
  # |letter|number|
  # +------+------+
  # |     a|     1|
  # |     b|     2|
  # |     c|     3|
  # +------+------+

  # Test with if_else
  df1 = df.withColumn("new_number",
                      ifelse(F.col("number") == 1, 
                             F.lit(10), 
                             F.lit(0)
                            )
                      )
  # Output
  # +------+------+----------+
  # |letter|number|new_number|
  # +------+------+----------+
  # |     a|     1|        10|
  # |     b|     2|         0|
  # |     c|     3|         0|
  # +------+------+----------+
  assert df1.collect() == ([Row(letter='a', number=1, new_number=10),
                            Row(letter='b', number=2, new_number=0),
                            Row(letter='c', number=3, new_number=0)]
                          )