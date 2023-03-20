import pytest
from tidypyspark.datasets import get_penguins_path
import tidypyspark.tidypyspark_class as ts
from tidypyspark._unexported_utils import _is_perfect_sublist

'''
# for local testing
import tidypyspark.tidypyspark_class as ts
from pyspark.sql import SparkSession 
import pyspark.sql.functions as F 
spark = SparkSession.builder.getOrCreate()
import pyspark
pen = spark.read.csv('src/tidypyspark/data/pen.csv', header = True).drop("_c0")
'''

@pytest.fixture
def penguins_data():
    return str(get_penguins_path())

def test_mutate(penguins_data):
  
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  res = (pen.ts.mutate({'bl_+_1': F.col('bill_length_mm') + 1,
                        'bl_+_1_by_2': F.col('bl_+_1') / 2}
                       )
         )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  cns = res.ts.colnames
  assert set(['bl_+_1', 'bl_+_1_by_2']).issubset(cns)
  
  spark.stop()

def test_rename(penguins_data):
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    spark = SparkSession.builder.getOrCreate()
    import pyspark
    pen = spark.read.csv(penguins_data, header=True).drop("_c0")
    res = pen.ts.rename({'species': 'species_2'})
    cns = res.ts.colnames
    assert isinstance(res, pyspark.sql.dataframe.DataFrame)
    assert set(['species_2']).issubset(cns)
    assert ['species'] not in cns
    spark.stop()

def test_relocate(penguins_data):
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    spark = SparkSession.builder.getOrCreate()
    import pyspark
    pen = spark.read.csv(penguins_data, header=True).drop("_c0")

    #testing after clause
    res = pen.ts.relocate(["island", "species"], after = "year")
    cns = res.ts.colnames
    assert isinstance(res, pyspark.sql.dataframe.DataFrame), "before clause failed 1"
    assert _is_perfect_sublist(["year", "island", "species"],  cns), "after clause failed 2"

    #testing before clause
    res = pen.ts.relocate(["island", "species"], before = "year")
    cns = res.ts.colnames
    assert isinstance(res, pyspark.sql.dataframe.DataFrame), "before clause failed 1"
    assert _is_perfect_sublist(["island", "species","year"],  cns), "before clause failed 2"

    #testing without before-after clause
    res = pen.ts.relocate(["bill_length_mm","bill_depth_mm"])
    cns = res.ts.colnames
    assert isinstance(res, pyspark.sql.dataframe.DataFrame), "without before-after clause failed 1"
    assert _is_perfect_sublist(["bill_length_mm","bill_depth_mm", "species"],  cns), "without before-after clause failed 2"
    spark.stop()

def test_summarise(penguins_data):
  
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  res = (pen.ts.mutate({'bl_+_1': F.col('bill_length_mm') + 1,
                        'bl_+_1_by_2': F.col('bl_+_1') / 2}
                       )
         )
  
  # ungrouped summarise
  res = (pen.ts.summarise({'mean_bl': F.mean(F.col('bill_length_mm')),
                           'count_species': F.count(F.col('species'))
                          }
                         )
            )
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == 1
  cns = res.ts.colnames
  assert set(['mean_bl', 'count_species']).issubset(cns)
  assert set(cns).issubset(['mean_bl', 'count_species'])
  
  
  # grouped summarise
  res = (pen.ts.summarize({'mean_bl': F.mean(F.col('bill_length_mm')),
                           'count_species': F.count(F.col('species'))
                          },
                          by = 'island'
                         )
            )
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == 3
  cns = res.ts.colnames
  assert set(['mean_bl', 'count_species', 'island']).issubset(cns)
  assert set(cns).issubset(['mean_bl', 'count_species', 'island'])
  
  spark.stop()
  
def test_count(penguins_data):
  
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  res = (pen.ts.mutate({'bl_+_1': F.col('bill_length_mm') + 1,
                        'bl_+_1_by_2': F.col('bl_+_1') / 2}
                       )
         )
  
  # count tests --------------------------------------------------------------
  # string input test
  res = pen.ts.count('species', name = 'cnt')
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == 3
  cns = res.ts.colnames
  assert set(['species', 'cnt']).issubset(cns)
  assert set(cns).issubset(['species', 'cnt'])
  
  # list input test
  res = pen.ts.count(['species', 'sex'])
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == 8
  cns = res.ts.colnames
  assert set(['species', 'sex', 'n']).issubset(cns)
  assert set(cns).issubset(['species', 'sex', 'n'])
  
  # test wt
  res = pen.ts.count('species', wt = 'body_mass_g')
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  
  
  # add count tests ----------------------------------------------------------
  # string input test
  res = pen.ts.add_count('species', name = 'cnt')
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == pen.count()
  cns = res.ts.colnames
  assert len(cns) == len(pen.ts.colnames) + 1
  assert set(['cnt']).issubset(cns)
  
  # list input test
  res = pen.ts.add_count(['species', 'sex'])
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res.count() == pen.count()
  cns = res.ts.colnames
  assert len(cns) == len(pen.ts.colnames) + 1
  assert set(['n']).issubset(cns)
  
  # test wt
  res = pen.ts.add_count('species', wt = 'body_mass_g')
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  
  spark.stop()

# join tests -----------------------------------------------------------------
def test_left_join_on_command(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 20), 
      (1, "SDE", 10),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # Perform Left Join using different combinations of on clause.
  left_join_df_1 = df1.ts.left_join(df2, on = ["id"])
  left_join_df_2 = df1.ts.left_join(df2, on = "id")

  # left_join_df_1 and left_join_df_2 should be the same. As follows:
  #   +---+------+----+------+---+
  # | id|  name|dept|dept_y|age|
  # +---+------+----+------+---+
  # |  1|jordan|  DS|   SDE| 10|
  # |  2|  jack|  DS|    BA| 40|
  # |  2|  jack|  DS|    PM| 30|
  # |  2|  jack|  DS|   SDE| 20|
  # |  1|  jack| SDE|   SDE| 10|
  # |  2|  jack| SDE|    BA| 40|
  # |  2|  jack| SDE|    PM| 30|
  # |  2|  jack| SDE|   SDE| 20|
  # |  1|  jack|  PM|   SDE| 10|
  # |  2|  jack|  PM|    BA| 40|
  # |  2|  jack|  PM|    PM| 30|
  # |  2|  jack|  PM|   SDE| 20|
  # +---+------+----+------+---+

  assert isinstance(left_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert left_join_df_1.count() == 12
  assert set(left_join_df_1.columns) == set(['id', 'name', 'dept', 'dept_y', 'age'])

  assert left_join_df_2.count() == 12
  assert set(left_join_df_2.columns) == set(['id', 'name', 'dept', 'dept_y', 'age'])

  left_join_df_3 = df1.ts.left_join(df2, on = ["id", "dept"])

  #   +---+----+------+----+
  # | id|dept|  name| age|
  # +---+----+------+----+
  # |  1|  DS|jordan|null|
  # |  2|  DS|  jack|null|
  # |  1| SDE|  jack|  10|
  # |  2| SDE|  jack|  20|
  # |  1|  PM|  jack|null|
  # |  2|  PM|  jack|  30|
  # +---+----+------+----+

  assert left_join_df_3.count() == 6
  assert set(left_join_df_3.columns) == set(['id', 'name', 'dept', 'age'])

  spark.stop()

def test_left_join_sql_on(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # Perform Left Join using different combinations of sql_on clause.
  left_join_df_1 = df1.ts.left_join(df2, sql_on = 'LHS.id == RHS.id')

  #   +---+------+----+----+------+---+
  # | id|  name|dept|id_y|dept_y|age|
  # +---+------+----+----+------+---+
  # |  1|jordan|  DS|   1|   SDE| 20|
  # |  2|  jack|  DS|   2|    BA| 40|
  # |  2|  jack|  DS|   2|    PM| 30|
  # |  2|  jack|  DS|   2|   SDE| 10|
  # |  1|  jack| SDE|   1|   SDE| 20|
  # |  2|  jack| SDE|   2|    BA| 40|
  # |  2|  jack| SDE|   2|    PM| 30|
  # |  2|  jack| SDE|   2|   SDE| 10|
  # |  1|  jack|  PM|   1|   SDE| 20|
  # |  2|  jack|  PM|   2|    BA| 40|
  # |  2|  jack|  PM|   2|    PM| 30|
  # |  2|  jack|  PM|   2|   SDE| 10|
  # +---+------+----+----+------+---+

  assert isinstance(left_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert left_join_df_1.count() == 12
  assert set(left_join_df_1.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  left_join_df_2 = df1.ts.left_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept)')
  # +---+------+----+----+------+----+
  # | id|  name|dept|id_y|dept_y| age|
  # +---+------+----+----+------+----+
  # |  1|jordan|  DS|null|  null|null|
  # |  2|  jack|  DS|null|  null|null|
  # |  1|  jack| SDE|   1|   SDE|  20|
  # |  2|  jack| SDE|   2|   SDE|  10|
  # |  1|  jack|  PM|null|  null|null|
  # |  2|  jack|  PM|   2|    PM|  30|
  # +---+------+----+----+------+----+
  assert left_join_df_2.count() == 6
  assert set(left_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  left_join_df_3 = df1.ts.left_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')
  # +---+------+----+----+------+----+
  # | id|  name|dept|id_y|dept_y| age|
  # +---+------+----+----+------+----+
  # |  1|jordan|  DS|null|  null|null|
  # |  2|  jack|  DS|null|  null|null|
  # |  1|  jack| SDE|   1|   SDE|  20|
  # |  2|  jack| SDE|   2|   SDE|  10|
  # |  1|  jack|  PM|null|  null|null|
  # |  2|  jack|  PM|null|  null|null|
  # +---+------+----+----+------+----+
  assert left_join_df_3.count() == 6
  assert set(left_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_left_join_on_x_on_y(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # left_join on x and y
  left_join_df_1 = df1.ts.left_join(df2, on_x = ['id', 'dept'], on_y = ['id', 'dept'])

  #   +---+----+------+----+
  # | id|dept|  name| age|
  # +---+----+------+----+
  # |  1|  DS|jordan|null|
  # |  2|  DS|  jack|null|
  # |  1| SDE|  jack|  20|
  # |  2| SDE|  jack|  10|
  # |  1|  PM|  jack|null|
  # |  2|  PM|  jack|  30|
  # +---+----+------+----+

  assert isinstance(left_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert left_join_df_1.count() == 6
  assert set(left_join_df_1.columns) == set(['id', 'dept', 'name', 'age'])

  left_join_df_2 = df1.ts.left_join(df2, on_x = ['name'], on_y = ['dept'])
  # +---+------+----+----+------+----+
  # | id|  name|dept|id_y|dept_y| age|
  # +---+------+----+----+------+----+
  # |  1|jordan|  DS|null|  null|null|
  # |  2|  jack|  DS|null|  null|null|
  # |  1|  jack| SDE|null|  null|null|
  # |  2|  jack| SDE|null|  null|null|
  # |  1|  jack|  PM|null|  null|null|
  # |  2|  jack|  PM|null|  null|null|
  # +---+------+----+----+------+----+
  assert left_join_df_2.count() == 6
  assert set(left_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  left_join_df_3 = df1.ts.left_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept'])
  # +---+------+----+----+------+----+
  # | id|  name|dept|id_y|dept_y| age|
  # +---+------+----+----+------+----+
  # |  1|jordan|  DS|null|  null|null|
  # |  2|  jack|  DS|null|  null|null|
  # |  1|  jack| SDE|null|  null|null|
  # |  2|  jack| SDE|null|  null|null|
  # |  1|  jack|  PM|null|  null|null|
  # |  2|  jack|  PM|null|  null|null|
  # +---+------+----+----+------+----+
  assert left_join_df_3.count() == 6
  assert set(left_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_inner_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # left_join on x and y
  inner_join_df_1 = df1.ts.inner_join(df2, on_x = ['id', 'dept'], on_y = ['id', 'dept'])

  #   +---+----+----+---+
  # | id|dept|name|age|
  # +---+----+----+---+
  # |  1| SDE|jack| 20|
  # |  2|  PM|jack| 30|
  # |  2| SDE|jack| 10|
  # +---+----+----+---+

  assert isinstance(inner_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert inner_join_df_1.count() == 3
  assert set(inner_join_df_1.columns) == set(['id', 'dept', 'name', 'age'])

  inner_join_df_2 = df1.ts.inner_join(df2, on_x = ['name'], on_y = ['dept'])
  # +---+----+----+----+------+---+
  # | id|name|dept|id_y|dept_y|age|
  # +---+----+----+----+------+---+
  # +---+----+----+----+------+---+
  assert inner_join_df_2.count() == 0
  assert set(inner_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  inner_join_df_3 = df1.ts.inner_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept'])
  # +---+----+----+----+------+---+
  # | id|name|dept|id_y|dept_y|age|
  # +---+----+----+----+------+---+
  # +---+----+----+----+------+---+
  assert inner_join_df_3.count() == 0
  assert set(inner_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_inner_join_sql_on(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # Perform Inner Join using different combinations of sql_on clause.
  inner_join_df_1 = df1.ts.inner_join(df2, sql_on = 'LHS.id == RHS.id')
  inner_join_df_2 = df1.ts.inner_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept)')
  inner_join_df_3 = df1.ts.inner_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')

  assert isinstance(inner_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert inner_join_df_1.count() == 12
  assert inner_join_df_2.count() == 3
  assert inner_join_df_3.count() == 2
  
  spark.stop()

def test_right_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # left_join on x and y
  right_join_df_1 = df1.ts.right_join(df2, on_x = ['id', 'dept'], on_y = ['id', 'dept'])

  # +---+----+----+---+
  # | id|dept|name|age|
  # +---+----+----+---+
  # |  2| SDE|jack| 10|
  # |  1| SDE|jack| 20|
  # |  2|  PM|jack| 30|
  # |  2|  BA|null| 40|
  # +---+----+----+---+

  assert isinstance(right_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert right_join_df_1.count() == 4
  assert set(right_join_df_1.columns) == set(['id', 'dept', 'name', 'age'])

  right_join_df_2 = df1.ts.right_join(df2, on_x = ['name'], on_y = ['dept'])
  # +----+----+----+----+------+---+
  # |  id|name|dept|id_y|dept_y|age|
  # +----+----+----+----+------+---+
  # |null|null|null|   2|   SDE| 10|
  # |null|null|null|   1|   SDE| 20|
  # |null|null|null|   2|    PM| 30|
  # |null|null|null|   2|    BA| 40|
  # +----+----+----+----+------+---+
  assert right_join_df_2.count() == 4
  assert set(right_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  right_join_df_3 = df1.ts.right_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept'])
  # +----+----+----+----+------+---+
  # |  id|name|dept|id_y|dept_y|age|
  # +----+----+----+----+------+---+
  # |null|null|null|   2|   SDE| 10|
  # |null|null|null|   1|   SDE| 20|
  # |null|null|null|   2|    PM| 30|
  # |null|null|null|   2|    BA| 40|
  # +----+----+----+----+------+---+
  assert right_join_df_3.count() == 4
  assert set(right_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_right_join_sql_on(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  right_join_df_2 = df1.ts.right_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept)')
  # +----+----+----+----+------+---+
  # |  id|name|dept|id_y|dept_y|age|
  # +----+----+----+----+------+---+
  # |   2|jack| SDE|   2|   SDE| 10|
  # |   1|jack| SDE|   1|   SDE| 20|
  # |   2|jack|  PM|   2|    PM| 30|
  # |null|null|null|   2|    BA| 40|
  # +----+----+----+----+------+---+
  assert right_join_df_2.count() == 4
  assert set(right_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  right_join_df_3 = df1.ts.right_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')
  # +----+----+----+----+------+---+
  # |  id|name|dept|id_y|dept_y|age|
  # +----+----+----+----+------+---+
  # |   2|jack| SDE|   2|   SDE| 10|
  # |   1|jack| SDE|   1|   SDE| 20|
  # |null|null|null|   2|    PM| 30|
  # |null|null|null|   2|    BA| 40|
  # +----+----+----+----+------+---+
  assert right_join_df_3.count() == 4
  assert set(right_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_full_join_sql_on(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  full_join_df_2 = df1.ts.full_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept)')
  # +----+------+----+----+------+----+
  # |  id|  name|dept|id_y|dept_y| age|
  # +----+------+----+----+------+----+
  # |   1|jordan|  DS|null|  null|null|
  # |   1|  jack|  PM|null|  null|null|
  # |   1|  jack| SDE|   1|   SDE|  20|
  # |null|  null|null|   2|    BA|  40|
  # |   2|  jack|  DS|null|  null|null|
  # |   2|  jack|  PM|   2|    PM|  30|
  # |   2|  jack| SDE|   2|   SDE|  10|
  # +----+------+----+----+------+----+
  assert full_join_df_2.count() == 7
  assert set(full_join_df_2.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  full_join_df_3 = df1.ts.full_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')
  # +----+------+----+----+------+----+
  # |  id|  name|dept|id_y|dept_y| age|
  # +----+------+----+----+------+----+
  # |   1|jordan|  DS|null|  null|null|
  # |   1|  jack|  PM|null|  null|null|
  # |   1|  jack| SDE|   1|   SDE|  20|
  # |null|  null|null|   2|    BA|  40|
  # |   2|  jack|  DS|null|  null|null|
  # |   2|  jack|  PM|null|  null|null|
  # |null|  null|null|   2|    PM|  30|
  # |   2|  jack| SDE|   2|   SDE|  10|
  # +----+------+----+----+------+----+
  assert full_join_df_3.count() == 8
  assert set(full_join_df_3.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_full_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # left_join on x and y
  full_join_df_1 = df1.ts.full_join(df2, on_x = ['id', 'dept'], on_y = ['id', 'dept'])

  # +---+----+------+----+
  # | id|dept|  name| age|
  # +---+----+------+----+
  # |  1|  DS|jordan|null|
  # |  1|  PM|  jack|null|
  # |  1| SDE|  jack|  20|
  # |  2|  BA|  null|  40|
  # |  2|  DS|  jack|null|
  # |  2|  PM|  jack|  30|
  # |  2| SDE|  jack|  10|
  # +---+----+------+----+

  assert isinstance(full_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert full_join_df_1.count() == 7
  assert set(full_join_df_1.columns) == set(['id', 'dept', 'name', 'age'])
  
  spark.stop()

def test_anti_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # anti_join on x and y
  anti_join_df_1 = df1.ts.anti_join(df2, on = ['id', 'dept'])

  # +---+----+------+
  # | id|dept|  name|
  # +---+----+------+
  # |  1|  DS|jordan|
  # |  2|  DS|  jack|
  # |  1|  PM|  jack|
  # +---+----+------+

  assert isinstance(anti_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert anti_join_df_1.count() == 3
  assert set(anti_join_df_1.columns) == set(['id', 'dept', 'name'])

  # anti_join on x and y
  anti_join_df_2 = df1.ts.anti_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept'])
  # +---+------+----+
  # | id|  name|dept|
  # +---+------+----+
  # |  1|jordan|  DS|
  # |  2|  jack|  DS|
  # |  1|  jack| SDE|
  # |  2|  jack| SDE|
  # |  1|  jack|  PM|
  # |  2|  jack|  PM|
  # +---+------+----+
  assert anti_join_df_2.count() == 6
  assert set(anti_join_df_2.columns) == set(['id', 'name', 'dept'])

  anti_join_df_3 = df1.ts.anti_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')
  # +---+------+----+
  # | id|  name|dept|
  # +---+------+----+
  # |  1|jordan|  DS|
  # |  2|  jack|  DS|
  # |  1|  jack|  PM|
  # |  2|  jack|  PM|
  # +---+------+----+
  assert anti_join_df_3.count() == 4
  assert set(anti_join_df_3.columns) == set(['id', 'name', 'dept'])
  
  spark.stop()

def test_semi_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # semi_join on x and y
  semi_join_df_1 = df1.ts.semi_join(df2, on = ['id', 'dept'])

  # +---+----+----+
  # | id|dept|name|
  # +---+----+----+
  # |  1| SDE|jack|
  # |  2|  PM|jack|
  # |  2| SDE|jack|
  # +---+----+----+

  assert isinstance(semi_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert semi_join_df_1.count() == 3
  assert set(semi_join_df_1.columns) == set(['id', 'dept', 'name'])

  # semi_join on x and y
  semi_join_df_2 = df1.ts.semi_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept'])
  # +---+----+----+
  # | id|name|dept|
  # +---+----+----+
  # +---+----+----+
  assert semi_join_df_2.count() == 0
  assert set(semi_join_df_2.columns) == set(['id', 'name', 'dept'])

  semi_join_df_3 = df1.ts.semi_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)')
  # +---+----+----+
  # | id|name|dept|
  # +---+----+----+
  # |  1|jack| SDE|
  # |  2|jack| SDE|
  # +---+----+----+
  assert semi_join_df_3.count() == 2
  assert set(semi_join_df_3.columns) == set(['id', 'name', 'dept'])
  
  spark.stop()

def test_cross_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # cross_join on x and y
  cross_join_df_1 = df1.ts.cross_join(df2)
  assert isinstance(cross_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert cross_join_df_1.count() == 24
  assert set(cross_join_df_1.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])
  
  spark.stop()

def test_join(penguins_data):

  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # Create the DataFrames
  df1 = spark.createDataFrame([
      (1, "jordan", 'DS'),
      (2, "jack", 'DS'),
      (1, "jack", 'SDE'),
      (2, "jack", 'SDE'),
      (1, "jack", 'PM'),
      (2, "jack", 'PM')
      ],        
      ("id", "name", "dept")
  )

  df2 = spark.createDataFrame([
      (2, "SDE", 10), 
      (1, "SDE", 20),
      (2, "PM", 30),
      (2, "BA", 40),
      ],
    ("id", "dept", "age")
  )

  # cross_join on x and y
  cross_join_df_1 = df1.ts.join(df2, how = 'cross')
  assert isinstance(cross_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert cross_join_df_1.count() == 24
  assert set(cross_join_df_1.columns) == set(['id', 'name', 'dept', 'id_y', 'dept_y', 'age'])

  # left_join on x and y
  left_join_df_1 = df1.ts.join(df2, on = ["id"] , how = 'left')
  assert isinstance(left_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert left_join_df_1.count() == 12

  # anti_join on x and y
  anti_join_df_1 = df1.ts.join(df2, on = ["id", "dept"] , how = 'anti')
  assert isinstance(anti_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert anti_join_df_1.count() == 3

  # semi_join on x and y
  semi_join_df_1 = df1.ts.join(df2, on = ["id", "dept"] , how = 'semi')
  assert isinstance(semi_join_df_1, pyspark.sql.dataframe.DataFrame)
  assert semi_join_df_1.count() == 3
  
  spark.stop()
  
def test_pipe_tee(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  res = pen.ts.pipe_tee(
    lambda x: x.select(['species', 'bill_length_mm']).show(6)
    )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert res == pen
  
  spark.stop()
  
def test_slice(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  # min ----------------------------------------------------------------------
  # with ties check
  res = pen.ts.slice_min(n = 2,
                         order_by_column = 'bill_depth_mm',
                         with_ties = True,
                         by = ['species', 'sex']
                         )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  # check if each group has atleast 2
  res  = pen.ts.count(['species', 'sex'])
  res2 = res.filter(F.col('n') >= F.lit(2))
  assert res.count() == res2.count()
  
  # without ties check
  res = pen.ts.slice_min(n = 2,
                         order_by_column = 'bill_depth_mm',
                         with_ties = False,
                         by = ['species', 'sex']
                         )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  # check if each group has exactly 2
  res  = res.ts.count(['species', 'sex'])
  res2 = res.filter(F.col('n') == F.lit(2))
  assert res.count() == res2.count()
  
  # max ----------------------------------------------------------------------
  # with ties check
  res = pen.ts.slice_max(n = 2,
                         order_by_column = 'bill_depth_mm',
                         with_ties = True,
                         by = ['species', 'sex']
                         )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  # check if each group has atleast 2
  res  = res.ts.count(['species', 'sex'])
  res2 = res.filter(F.col('n') >= F.lit(2))
  assert res.count() == res2.count()
  
  # without ties check
  res = pen.ts.slice_max(n = 2,
                         order_by_column = 'bill_depth_mm',
                         with_ties = False,
                         by = ['species', 'sex']
                         )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  # check if each group has exactly 2
  res  = res.ts.count(['species', 'sex'])
  res2 = res.filter(F.col('n') == F.lit(2))
  assert res.count() == res2.count()
  
  spark.stop()

def test_rbind(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  df1 = pen.select(['species', 'island', 'bill_length_mm'])
  df2 = pen.select(['species', 'island', 'bill_depth_mm'])
  
  res = df1.ts.rbind(df2)
  assert len(res.columns) == 4
  assert res.count() == (344 * 2)
  
  res = df1.ts.rbind(df2, id = "id")
  assert len(res.columns) == 5
  assert res.count() == (344 * 2)
  assert res.ts.count('id').count() == 2
  
  spark.stop()
  
def test_union(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  df1 = pen.ts.slice_max(n = 2,
                         order_by_column = 'bill_length_mm',
                         with_ties = False
                         )
  df2 = pen.ts.slice_max(n = 4,
                         order_by_column = 'bill_length_mm',
                         with_ties = False
                         )
  
  assert df1.ts.union(df2).count() == 4
  assert (df1.ts.union(df2.ts.select(['bill_length_mm', 'species']))
             .count()) == 6
  assert len(df1.ts.union(df2).columns) == len(df1.columns)
  
  spark.stop()

def test_pivot_longer():
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  #Create spark dataframe.
  data = [("Banana",1000,"USA"), ("Carrots",1500,"USA"), ("Beans",1600,"USA"), \
        ("Orange",2000,"USA"),("Orange",2000,"USA"),("Banana",400,"China"), \
        ("Carrots",1200,"China"),("Beans",1500,"China"),("Orange",4000,"China"), \
        ("Banana",2000,"Canada"),("Carrots",2000,"Canada"),("Beans",2000,"Mexico")]
  columns= ["Product","Amount","Country"]
  df = spark.createDataFrame(data = data, schema = columns)
  pivotDF = df.groupBy("Product").pivot("Country").sum("Amount")

  # +-------+------+-----+------+----+
  # |Product|Canada|China|Mexico|USA |
  # +-------+------+-----+------+----+
  # |Orange |null  |4000 |null  |4000|
  # |Beans  |null  |1500 |2000  |1600|
  # |Banana |2000  |400  |null  |1000|
  # |Carrots|2000  |1200 |null  |1500|
  # +-------+------+-----+------+----+

  # Test the pivot_longer function
  df1 = pivotDF.ts.pivot_longer(names_to = 'Country',
                                values_to = 'Total',
                                cols = ['Canada', 'China', 'Mexico'],
                                include = True,
                                )
  # Output
  # +----+-------+-------+-----+
  # | USA|Product|Country|Total|
  # +----+-------+-------+-----+
  # |4000| Orange| Canada| null|
  # |4000| Orange|  China| 4000|
  # |4000| Orange| Mexico| null|
  # |1600|  Beans| Canada| null|
  # |1600|  Beans|  China| 1500|
  # |1600|  Beans| Mexico| 2000|
  # |1000| Banana| Canada| 2000|
  # |1000| Banana|  China|  400|
  # |1000| Banana| Mexico| null|
  # |1500|Carrots| Canada| 2000|
  # |1500|Carrots|  China| 1200|
  # |1500|Carrots| Mexico| null|
  # +----+-------+-------+-----+

  assert isinstance(df1, pyspark.sql.dataframe.DataFrame)
  assert df1.count() == 12
  assert set(df1.columns) == set(['USA', 'Product', 'Country', 'Total'])

  # Test the pivot_longer function
  df2 = pivotDF.ts.pivot_longer(names_to = 'Country',
                                values_to = 'Total',
                                cols = ['Canada'],
                                include = True
                               )
  # Output
  # +-----+----+-------+------+-------+-----+
  # |China| USA|Product|Mexico|Country|Total|
  # +-----+----+-------+------+-------+-----+
  # | 4000|4000| Orange|  null| Canada| null|
  # | 1500|1600|  Beans|  2000| Canada| null|
  # |  400|1000| Banana|  null| Canada| 2000|
  # | 1200|1500|Carrots|  null| Canada| 2000|
  # +-----+----+-------+------+-------+-----+

  assert isinstance(df2, pyspark.sql.dataframe.DataFrame)
  assert df2.count() == 4
  assert set(df2.columns) == set(['China', 'USA', 'Product', 'Mexico', 'Country', 'Total'])

  # Test the pivot_longer function with values_drop_na = True
  # Test the pivot_longer function
  df3 = pivotDF.ts.pivot_longer(names_to = 'Country',
                                values_to = 'Total',
                                cols = ['Canada'],
                                include = True,
                                values_drop_na = True
                               )
  
  # Output
  # +----+-------+-----+------+-------+-----+
  # | USA|Product|China|Mexico|Country|Total|
  # +----+-------+-----+------+-------+-----+
  # |1000| Banana|  400|  null| Canada| 2000|
  # |1500|Carrots| 1200|  null| Canada| 2000|
  # +----+-------+-----+------+-------+-----+

  assert df3.count() == 2
  assert set(df3.columns) == set(['USA', 'Product', 'China', 'Mexico', 'Country', 'Total'])
  
  spark.stop()

def test_pivot_wider():
  import pyspark.sql.functions as F
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  #Create spark dataframe.
  df = spark.createDataFrame([(1, "jordan", 'DS'),
                              (2, "jack", 'DS'),
                              (1, "jack", 'SDE'),
                              (2, "jack", 'SDE'),
                              (1, "jack", 'PM'),
                              (2, "jack", 'PM')
                              ],
                              ("id", "name", "dept")
                          )
  df = (df.withColumn('name2', F.concat_ws("", F.col('name'), F.lit("_2")))
          .withColumn('name3', F.concat_ws("", F.col('name'), F.lit("_3")))
          .withColumn('dept2', F.concat_ws("", F.col('dept'), F.lit("_2")))
          .withColumn('dept3', F.concat_ws("", F.col('dept'), F.lit("_3")))
      )
  
  # Input dataframe
  # +---+------+----+--------+--------+-----+-----+
  # | id|  name|dept|   name2|   name3|dept2|dept3|
  # +---+------+----+--------+--------+-----+-----+
  # |  1|jordan|  DS|jordan_2|jordan_3| DS_2| DS_3|
  # |  2|  jack|  DS|  jack_2|  jack_3| DS_2| DS_3|
  # |  1|  jack| SDE|  jack_2|  jack_3|SDE_2|SDE_3|
  # |  2|  jack| SDE|  jack_2|  jack_3|SDE_2|SDE_3|
  # |  1|  jack|  PM|  jack_2|  jack_3| PM_2| PM_3|
  # |  2|  jack|  PM|  jack_2|  jack_3| PM_2| PM_3|
  # +---+------+----+--------+--------+-----+-----+

  # Test the pivot_wider function
  df1 = df.ts.pivot_wider(id_cols = "id",
                          names_from = ['name', 'name2', 'name3'],
                          values_from = "dept",
                          names_expand = True
                          )
  # Output
  # ---+--------------------------+------------------------+------------------------+----------------------+------------------------+----------------------+----------------------+--------------------+
  # | id|jordan__jordan_2__jordan_3|jack__jordan_2__jordan_3|jordan__jack_2__jordan_3|jack__jack_2__jordan_3|jordan__jordan_2__jack_3|jack__jordan_2__jack_3|jordan__jack_2__jack_3|jack__jack_2__jack_3|
  # +---+--------------------------+------------------------+------------------------+----------------------+------------------------+----------------------+----------------------+--------------------+
  # |  1|                      [DS]|                    null|                    null|                  null|                    null|                  null|                  null|           [SDE, PM]|
  # |  2|                      null|                    null|                    null|                  null|                    null|                  null|                  null|       [DS, SDE, PM]|
  # +---+--------------------------+------------------------+------------------------+----------------------+------------------------+----------------------+----------------------+--------------------+
  assert isinstance(df1, pyspark.sql.dataframe.DataFrame)
  assert df1.count() == 2
  assert len(df1.columns) == 9

  # Test the pivot_wider function
  df2 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name', 'name2'],
                          values_from = ['dept', 'dept2'],
                          names_expand = True
                          )
  # Output
  # +---+-----+---------------------+----------------------+-------------------+--------------------+-------------------+--------------------+-----------------+------------------+
  # | id|dept3|jordan__jordan_2_dept|jordan__jordan_2_dept2|jack__jordan_2_dept|jack__jordan_2_dept2|jordan__jack_2_dept|jordan__jack_2_dept2|jack__jack_2_dept|jack__jack_2_dept2|
  # +---+-----+---------------------+----------------------+-------------------+--------------------+-------------------+--------------------+-----------------+------------------+
  # |  1| DS_3|                   DS|                  DS_2|               null|                null|               null|                null|             null|              null|
  # |  2| DS_3|                 null|                  null|               null|                null|               null|                null|               DS|              DS_2|
  # |  1|SDE_3|                 null|                  null|               null|                null|               null|                null|              SDE|             SDE_2|
  # |  2|SDE_3|                 null|                  null|               null|                null|               null|                null|              SDE|             SDE_2|
  # |  1| PM_3|                 null|                  null|               null|                null|               null|                null|               PM|              PM_2|
  # |  2| PM_3|                 null|                  null|               null|                null|               null|                null|               PM|              PM_2|
  # +---+-----+---------------------+----------------------+-------------------+--------------------+-------------------+--------------------+-----------------+------------------+
  assert df2.count() == 6
  assert len(df2.columns) == 10

  # Test the pivot_wider function with names_expand = False (default value)
  df3 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name', 'name2'],
                          values_from = ['dept', 'dept2'],
                         )
  # Output
  # +---+-----+---------------------+----------------------+-----------------+------------------+
  # | id|dept3|jordan__jordan_2_dept|jordan__jordan_2_dept2|jack__jack_2_dept|jack__jack_2_dept2|
  # +---+-----+---------------------+----------------------+-----------------+------------------+
  # |  1| DS_3|                   DS|                  DS_2|             null|              null|
  # |  2| DS_3|                 null|                  null|               DS|              DS_2|
  # |  1|SDE_3|                 null|                  null|              SDE|             SDE_2|
  # |  2|SDE_3|                 null|                  null|              SDE|             SDE_2|
  # |  1| PM_3|                 null|                  null|               PM|              PM_2|
  # |  2| PM_3|                 null|                  null|               PM|              PM_2|
  # +---+-----+---------------------+----------------------+-----------------+------------------+
  assert df3.count() == 6
  assert len(df3.columns) == 6

  # Test the pivot_wider function with values_fn = "F.first" and names_prefix = "new_col"
  df4 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name', 'name2'],
                          values_from = ['dept', 'dept2'],
                          values_fn = "F.first",
                          names_prefix = "new_col"
                          )
  # Output
  # +---+-----+------------------------------+-------------------------------+--------------------------+---------------------------+
  # | id|dept3|new_col__jordan__jordan_2_dept|new_col__jordan__jordan_2_dept2|new_col__jack__jack_2_dept|new_col__jack__jack_2_dept2|
  # +---+-----+------------------------------+-------------------------------+--------------------------+---------------------------+
  # |  1| DS_3|                            DS|                           DS_2|                      null|                       null|
  # |  1| PM_3|                          null|                           null|                        PM|                       PM_2|
  # |  1|SDE_3|                          null|                           null|                       SDE|                      SDE_2|
  # |  2| DS_3|                          null|                           null|                        DS|                       DS_2|
  # |  2| PM_3|                          null|                           null|                        PM|                       PM_2|
  # |  2|SDE_3|                          null|                           null|                       SDE|                      SDE_2|
  # +---+-----+------------------------------+-------------------------------+--------------------------+---------------------------+
  assert df4.count() == 6
  assert len(df4.columns) == 6
  assert set(df4.columns) == set(['id', 'dept3', 'new_col__jordan__jordan_2_dept', 'new_col__jordan__jordan_2_dept2',
                                  'new_col__jack__jack_2_dept', 'new_col__jack__jack_2_dept2'])
  
  # Test the pivot_wider function with values_fn = {'dept': 'F.first', 'dept2': 'F.last'}
  df5 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name'],
                          values_from = ['dept', 'dept2'],
                          values_fn = {'dept': 'F.first', 'dept2': 'F.last'}
                          )
  # Output
  # +---+-----+-----------+------------+---------+----------+
  # | id|dept3|jordan_dept|jordan_dept2|jack_dept|jack_dept2|
  # +---+-----+-----------+------------+---------+----------+
  # |  1| DS_3|         DS|        DS_2|     null|      null|
  # |  1| PM_3|       null|        null|       PM|      PM_2|
  # |  1|SDE_3|       null|        null|      SDE|     SDE_2|
  # |  2| DS_3|       null|        null|       DS|      DS_2|
  # |  2| PM_3|       null|        null|       PM|      PM_2|
  # |  2|SDE_3|       null|        null|      SDE|     SDE_2|
  # +---+-----+-----------+------------+---------+----------+
  assert df5.count() == 6
  assert len(df5.columns) == 6
  assert set(df5.columns) == set(['id', 'dept3', 'jordan_dept', 'jordan_dept2', 'jack_dept', 'jack_dept2'])
  
  # Test the pivot_wider function with values_fill = {'dept': 3, 'dept2': []} and values_fn = {'dept': 'F.first'}
  df6 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name'],
                          values_from = ['dept', 'dept2'],
                          values_fill = {'dept': 3, 'dept2': []},
                          values_fn = {'dept': 'F.first'},
                          )
  
  # Output
  # +---+-----+-----------+------------+---------+----------+
  # | id|dept3|jordan_dept|jordan_dept2|jack_dept|jack_dept2|
  # +---+-----+-----------+------------+---------+----------+
  # |  1| DS_3|         DS|      [DS_2]|        3|        []|
  # |  2| DS_3|          3|          []|       DS|    [DS_2]|
  # |  1|SDE_3|          3|          []|      SDE|   [SDE_2]|
  # |  2|SDE_3|          3|          []|      SDE|   [SDE_2]|
  # |  1| PM_3|          3|          []|       PM|    [PM_2]|
  # |  2| PM_3|          3|          []|       PM|    [PM_2]|
  # +---+-----+-----------+------------+---------+----------+
  assert df6.count() == 6
  assert len(df6.columns) == 6
  assert set(df6.columns) == set(['id', 'dept3', 'jordan_dept', 'jordan_dept2', 'jack_dept', 'jack_dept2'])
  
  # Test the pivot_wider function with values_fill = []
  df7 = df.ts.pivot_wider(id_cols = ["id", "dept3"],
                          names_from = ['name'],
                          values_from = ['dept', 'dept2'],
                          values_fill = []
                          )
  # Output
  # +---+-----+-----------+------------+---------+----------+
  # | id|dept3|jordan_dept|jordan_dept2|jack_dept|jack_dept2|
  # +---+-----+-----------+------------+---------+----------+
  # |  1| DS_3|       [DS]|      [DS_2]|       []|        []|
  # |  2| DS_3|         []|          []|     [DS]|    [DS_2]|
  # |  1|SDE_3|         []|          []|    [SDE]|   [SDE_2]|
  # |  2|SDE_3|         []|          []|    [SDE]|   [SDE_2]|
  # |  1| PM_3|         []|          []|     [PM]|    [PM_2]|
  # |  2| PM_3|         []|          []|     [PM]|    [PM_2]|
  # +---+-----+-----------+------------+---------+----------+
  assert df7.count() == 6
  assert len(df7.columns) == 6
  assert set(df7.columns) == set(['id', 'dept3', 'jordan_dept', 'jordan_dept2', 'jack_dept', 'jack_dept2'])

  spark.stop()
  
def test_nest_by(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  res = pen.ts.nest_by(by = ['species', 'island'])
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert len(res.columns) == 3
  assert res.ts.types['data'] == "array"
  
  spark.stop()
  
def test_unnest_wider(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  res = (pen.ts.nest_by(by = ['species', 'island'])
            .withColumn('data_exploded', F.explode('data'))
            .drop('data')
            )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert len(res.columns) == 3
  assert res.ts.types['data_exploded'] == "struct"
  
  spark.stop()
  
def test_unnest_longer(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  res = (pen.ts.nest_by(by = ['species', 'island'])
            .withColumn('data_exploded', F.explode('data'))
            .drop('data')
            .ts.unnest_longer('data_exploded')
            )
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert len(res.columns) == 4
  expected_colnames = ['species', 'island', 'name', 'value']
  assert all([x in res.columns for x in expected_colnames])
  
  spark.stop()

def test_unnest(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  # unnest check on nested structs
  res  = pen.ts.nest_by(by = ['species', 'island'])
  res2 = res.ts.unnest('data')
  
  assert isinstance(res, pyspark.sql.dataframe.DataFrame)
  assert pen.ts.types == res2.ts.types
  
  # unnest check on arrays
  res = (pen.groupby('species')
            .agg(F.collect_list('year').alias('year_array'))
            )
  res2 = res.ts.unnest('year_array').withColumnRenamed('year_array', 'year')
  
  assert isinstance(res2, pyspark.sql.dataframe.DataFrame)
  assert pen.select('species', 'year').ts.types == res2.ts.types
  
  spark.stop()
  
def test_fill(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  data = [(None, None, 1, 1), 
          (1, None, 2, 2), 
          (1, 1, 1,3), 
          (2, 2, 2,4), 
          (2, None, 1,5), 
          (3, None, 2,6), 
          (None, 3, 1,7) 
          ] 
  df = spark.createDataFrame(data, ["A", "B", "C", "rowid"])
  
  # down
  res = df.ts.fill_na({"A": "down"}, order_by = "rowid").ts.to_list('A')
  exp = [None, 1,1,2,2,3,3]
  assert res ==  exp
  
  # up
  res = df.ts.fill_na({"A": "up"}, order_by = "rowid").ts.to_list('A')
  exp = [1, 1,1,2,2,3,None]
  assert res ==  exp
  
  # updown
  res = df.ts.fill_na({"A": "updown"}, order_by = "rowid").ts.to_list('A')
  exp = [1,1,1,2,2,3,3]
  assert res ==  exp
  
  # downup
  res = df.ts.fill_na({"B": "downup"}, order_by = "rowid").ts.to_list('B')
  exp = [1,1,1,2,2,2,3]
  assert res ==  exp
  
  # grouped up
  res = (df.ts.fill_na({"B": "up"}, order_by = "rowid", by = "C")
           .orderBy('rowid')
           .ts.to_list('B')
           )
  exp = [1,2,1,2,3, None, 3]
  assert res ==  exp
  
  # grouped down
  res = (df.ts.fill_na({"B": "down"}, order_by = "rowid", by = "C")
           .orderBy('rowid')
           .ts.to_list('B')
           )
  exp = [None, None, 1,2, 1, 2, 3]
  assert res ==  exp
  
  spark.stop()
  
def test_to(penguins_data):
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  pen = spark.read.csv(penguins_data, header = True).drop("_c0")
  
  res = pen.ts.to_list('species')
  assert isinstance(res, list)
  
  res = pen.ts.pull('species')
  assert str(res.__class__.__name__) == "Series"
  
  res = pen.ts.to_dict()
  assert isinstance(res, dict)
  
  res = pen.ts.to_pandas()
  assert str(res.__class__.__name__) == "DataFrame"
  
  spark.stop()

def test_drop_na():
  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark
  
  from pyspark.sql.functions import col

  # create a DataFrame with null values
  data = [("Alice", 25, None), ("Bob", None, 80), (None, 30, 90)]
  df = spark.createDataFrame(data, ["name", "age", "score"])

  # drop rows with null values
  df1 = df.ts.drop_na()
  # Output
  # +----+---+-----+
  # |name|age|score|
  # +----+---+-----+
  # +----+---+-----+
  assert df1.count() == 0

  # drop rows with null values in a specific column
  df2 = df.ts.drop_na(subset = ["age"])
  # Output
  # +-----+---+-----+
  # | name|age|score|
  # +-----+---+-----+
  # |Alice| 25| null|
  # | null| 30|   90|
  # +-----+---+-----+
  assert df2.count() == 2

  # drop rows with null values if all values are null.
  df3 = df.ts.drop_na(how = "all")
  # Output
  # +-----+----+-----+
  # | name| age|score|
  # +-----+----+-----+
  # |Alice|  25| null|
  # |  Bob|null|   80|
  # | null|  30|   90|
  # +-----+----+-----+
  assert df3.count() == 3

  # drop rows with less than 3 non-null values
  df4 = df.ts.drop_na(thresh=3)
  # Output
  # +----+---+-----+
  # |name|age|score|
  # +----+---+-----+
  # +----+---+-----+
  assert df4.count() == 0
  
  spark.stop()

def test_replace_na():

  from pyspark.sql import SparkSession 
  import pyspark.sql.functions as F 
  spark = SparkSession.builder.getOrCreate()
  import pyspark

  # create a DataFrame with null values
  data = [("Alice", 25, None, [20, 30, 40]), 
          ("Bob", None, 80, [10, 20, 30]), 
          (None, 30, 90 , None)
          ]
  df = spark.createDataFrame(data, ["name", "age", "score", "marks"])
  # +-----+----+-----+------------+
  # | name| age|score|       marks|
  # +-----+----+-----+------------+
  # |Alice|  25| null|[20, 30, 40]|
  # |  Bob|null|   80|[10, 20, 30]|
  # | null|  30|   90|        null|
  # +-----+----+-----+------------+

  # replace null values with a scalar value
  df1 = df.ts.replace_na(0)
  # Output
  # +-----+---+-----+------------+
  # | name|age|score|       marks|
  # +-----+---+-----+------------+
  # |Alice| 25|    0|[20, 30, 40]|
  # |  Bob|  0|   80|[10, 20, 30]|
  # | null| 30|   90|        null|
  # +-----+---+-----+------------+
  assert df1.select("age").collect()[1][0] == 0
  assert df1.select("score").collect()[0][0] == 0

  # replace null values with a dictionary of column names and values
  df2 = df.ts.replace_na({"name": "A", "score": 25, "marks": []})
  # Output
  # +-----+----+-----+------------+
  # | name| age|score|       marks|
  # +-----+----+-----+------------+
  # |Alice|  25|   25|[20, 30, 40]|
  # |  Bob|null|   80|[10, 20, 30]|
  # |    A|  30|   90|          []|
  # +-----+----+-----+------------+
  assert df2.select("name").collect()[2][0] == "A"
  assert df2.select("marks").collect()[2][0] == []

  # replace null values in a specific column. Note we have used an empty list in the dictionary for column "marks"
  df3 = df.ts.replace_na({"age": True, "marks": []}, subset=["age", "score", "marks"])
  # +-----+---+-----+------------+
  # | name|age|score|       marks|
  # +-----+---+-----+------------+
  # |Alice| 25| null|[20, 30, 40]|
  # |  Bob|  1|   80|[10, 20, 30]|
  # | null| 30|   90|          []|
  # +-----+---+-----+------------+
  assert df3.select("age").collect()[1][0] == 1
  assert df3.select("marks").collect()[2][0] == []

  # replace null values in all columns with an empty list. 
  # This replaces null values in all columns of type array<> with an empty list
  df4 = df.ts.replace_na([], subset = None)
  # +-----+----+-----+------------+
  # | name| age|score|       marks|
  # +-----+----+-----+------------+
  # |Alice|  25| null|[20, 30, 40]|
  # |  Bob|null|   80|[10, 20, 30]|
  # | null|  30|   90|          []|
  # +-----+----+-----+------------+
  assert df4.select("marks").collect()[2][0] == []

  spark.stop()