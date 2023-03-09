import pytest
from tidypyspark.datasets import get_penguins_path
import tidypyspark.tidypyspark_class as ts
from tidypyspark._unexported_utils import _is_perfect_sublist

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