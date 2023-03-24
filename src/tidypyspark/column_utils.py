import pyspark.sql.functions as F

def ifelse(condition, yes, no):
  
  '''
  Vectorized if and else statement.
  ifelse returns a value with the same shape as condition which is filled with 
  elements selected from either yes or no depending on whether the element of
  condition is TRUE or FALSE.

  Parameters
  ----------
  condition: expression or pyspark col
      Should evaluate to a boolean list/array/Series
  yes: expression or list/array/Series
      Should evaluate to a pyspark col for true elements of condition.
  no: expression or list/array/Series
      Should evaluate to a pyspark col for false elements of condition.

  Returns
  -------
  pyspark col

  Examples
  --------
  >>> from pyspark.sql import SparkSession 
  >>> import pyspark.sql.functions as F 
  >>> spark = SparkSession.builder.getOrCreate()
  >>> import pyspark

  >>> df = spark.createDataFrame([("a", 1), ("b", 2), ("c", 3)],
                              ["letter", "number"]
                            )
  >>> df.show()
  +------+------+
  |letter|number|
  +------+------+
  |     a|     1|
  |     b|     2|
  |     c|     3|
  +------+------+

  >>> df.withColumn("new_number",
                    ifelse(F.col("number") == 1, 
                           F.lit(10), 
                           F.lit(0)
                           )
                    ).show()
  +------+------+----------+
  |letter|number|new_number|
  +------+------+----------+
  |     a|     1|        10|
  |     b|     2|         0|
  |     c|     3|         0|
  +------+------+----------+
  '''

  return F.when(condition, yes).otherwise(no)

def case_when(list_of_tuples, default = None):

  """
  Implements a case_when function using PySpark.
  
  Parameters:
  ----------
    list_of_tuples (list): 
      A list of tuples, where each tuple represents a condition 
      and its corresponding value.
    default (optional): 
    The default value to use when no conditions are met. Defaults to None.
      
  Returns:
  ----------
    PySpark Column: A PySpark column representing the case_when expression.

  Examples:
  ----------
  >>> from pyspark.sql import SparkSession 
  >>> import pyspark.sql.functions as F 
  >>> spark = SparkSession.builder.getOrCreate()
  >>> import pyspark

  >>> df = spark.createDataFrame([("a", 1), ("b", 2), ("c", 3)], 
                                 ["letter", "number"]
                                )
  >>> df.show()
  +------+------+
  |letter|number|
  +------+------+
  |     a|     1|
  |     b|     2|
  |     c|     3|
  +------+------+

  >>> df.withColumn("new_number",
                case_when([(F.col("number") == 1, F.lit(10)),
                            (F.col("number") == 1, F.lit(20)),
                            (F.col("number") == 3, F.lit(30))],
                            default = F.lit(0)
                          )
                ).show()
  +------+------+----------+
  |letter|number|new_number|
  +------+------+----------+
  |     a|     1|        10|
  |     b|     2|         0|
  |     c|     3|        30|
  +------+------+----------+

  """
  
  assert isinstance(list_of_tuples, list), \
    "list_of_tuples should be a list of tuples"

  assert all([isinstance(i, tuple) for i in list_of_tuples]),\
    "list_of_tuples should be a list of tuples"

  assert all([len(i) == 2 for i in list_of_tuples]),\
    "list_of_tuples should be a list of tuples of length 2"
  
  # Create a list of PySpark expressions for each condition in list_of_tuples
  conditions = ([F.when(condition,value) 
                for condition,value in list_of_tuples]
               )
  
  # Define a pyspark expression that checks conditions in order and returns
  # the corresponding value if the condition is met. If no conditions are met,
  # return the default value.
  if default is None:
    case_when_expression = F.coalesce(*conditions)
  else:
    case_when_expression = F.coalesce(*conditions, default)
  
  return case_when_expression