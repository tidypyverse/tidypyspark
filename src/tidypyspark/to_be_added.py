
# def test_to_pandas(penguins_data):
#     from pyspark.sql import SparkSession
#     import pyspark.sql.functions as F
#     spark = SparkSession.builder.getOrCreate()
#     import pyspark
#     # import pandas
#     pen = spark.read.csv(penguins_data, header=True).drop("_c0")
#     res = pen.ts.to_pandas()
#     assert isinstance(res, pandas.core.frame.DataFrame)
#
#     spark.stop()

# def to_pandas(self):
#   '''
#   to_pandas
#   Converts spark dataframe to pandas dataframe
#
#   Parameters
#   ----------
#   None
#
#   Examples
#   --------
#   (pen.ts.to_pandas()
#   .show(10)
#   )

##setup code
'''
from tidypyspark.datasets import get_penguins_path
import tidypyspark.tidypyspark_class as ts
from pyspark.sql import SparkSession
#Create PySpark SparkSession
spark = SparkSession.builder \
            .master("local[1]") \
            .appName("SparkByExamples.com") \
            .getOrCreate()
path_peng = get_penguins_path()
pen = spark.read.csv(path_peng, header = True).drop('_c0')


'''

#
#   '''
#   return self.__data.toPandas()
