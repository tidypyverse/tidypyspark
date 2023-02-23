import pytest
from tidypyspark.datasets import get_penguins_path
import tidypyspark.tidypyspark_class as ts


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