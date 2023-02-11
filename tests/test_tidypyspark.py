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