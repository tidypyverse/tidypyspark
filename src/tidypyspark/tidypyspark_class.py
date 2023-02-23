# ----------------------------------------------------------------------------
# This file is a part of tidypyspark python package
# Find the dev version here: https://github.com/talegari/tidypyspark
# ------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------
import warnings
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from tidypyspark.accessor_class import register_dataframe_accessor
import numpy as np
from tidypyspark._unexported_utils import (
                              _is_kwargable,
                              _is_valid_colname,
                              _is_string_or_string_list,
                              _enlist,
                              _get_unique_names,
                              _is_unique_list,
                              _generate_new_string,
                              _is_nested,
                              _flatten_strings,
                              _nested_is_unique
                            )

# tidypyspark ----
@register_dataframe_accessor('ts')
class acc_on_pyspark():
  
  def __init__(self, data):
    
    colnames = list(data.columns)
    
    assert all([_is_valid_colname(x) for x in colnames]),\
      "column names should not start with underscore"
    
    assert _is_unique_list(colnames),\
      "column names should be unique"
    
    self.__data = data
  
  # attributes -------------------------------------------------------------
  
  @property
  def nrow(self):
    print("Please run: `df.ts.count()` to get the number of rows")
    return None
  
  @property
  def ncol(self):
    return len(self.__data.columns)
  
  @property
  def colnames(self):
    return list(self.__data.columns)
  
  @property
  def shape(self):
    return (self.nrow, self.ncol)
  
  @property
  def dim(self):
    return (self.nrow, self.ncol)

# cleaners --------------------------------------------------------------
  
  def _clean_by(self, by):
    '''
    _clean_by
    validates by, cleans 'by' and returns a list of column names
    
    Parameters
    ----------
    by : string, list of strings

    Returns
    -------
    list of column names
    '''
    
    cns = self.colnames
    by = _enlist(by)
    
    assert _is_string_or_string_list(by),\
      "'by' should be a string or list of strings"
    assert _is_unique_list(by),\
      "'by' should be a list of unique strings"
    assert set(by).issubset(cns),\
      "'by' should be a list of valid column names"
    
    return by

  def _clean_order_by(self, order_by):
    '''
    _clean_order_by
    validates order_by, cleans and returns a list of 'column' objects
    
    Parameters
    ----------
    order_by : string or tuple or list of tuples
        Order by specification, see Notes

    Returns
    -------
    list of 'column' objects
    
    Notes
    -----
    1. A 'column' object is an instance of 'pyspark.sql.Column'.
    2. Prototype of return objects:
      - ["col_a", ("col_b", "desc")] --> [col('col_a'), col('col_b').desc()]
    '''
    
    cns = self.colnames
    
    # convert string to list, tuple to list
    order_by = _enlist(order_by)
    
    # handle order_by
    for id, x in enumerate(order_by):           
      
      # case 1: tuple
      # prototypes:
      # ('col_a', 'asc')
      # (F.col('col_a'), 'desc')
      if isinstance(x, tuple):
        assert len(x) == 2,\
          (f'Input tuple should have length 2. '
           f'Input: {x}'
           )
        assert isinstance(x[0], str),\
          (f'First element of the input tuple should be a string. '
           f'Input: {x}'
           )
        if x[1] not in ['asc', 'desc']:
          raise Exception((f'Second element of the input tuple should '
                           f'one among: ["asc", "desc"]. '
                           f'Input: {x}'
                           ))
        
        assert x[0] in cns,\
          (f'String input to order_by should be a valid column name. '
           f'Input: {x[0]}'
           )
          
        if x[1] == 'asc':
          order_by[id] = F.col(x[0]).asc()
        else:
          order_by[id] = F.col(x[0]).desc()
      
      # case 2: string
      # prototypes:
      # 'col_a'
      elif isinstance(x, str):
        assert x in cns,\
          (f'String input to order_by should be a valid column name. '
           f'Input: {x}'
           )
        order_by[id] = F.col(x)
      
      else:
        raise Exception(("An element of 'order_by' should be a tuple or "
                         "a string"))
      
    return order_by
      
  def _clean_column_names(self, column_names):
    '''
    _clean_column_names
    validates and returns cleaned column names
    
    Parameters
    ----------
    column_names : string or list of strings
      columns names to be cleaned

    Returns
    -------
    list of column names
    '''
    cns = self.colnames
    column_names = _enlist(column_names)
    
    assert _is_string_or_string_list(column_names),\
      "'column_names' should be a string or list of strings"
    assert _is_unique_list(column_names),\
      "'column_names' should be a list of unique strings"
    assert set(column_names).issubset(cns),\
      "'column_names' should be a list of valid column names"
    
    return column_names
  
  def _extract_order_by_cols(self, order_by):
    '''
    _extract_order_by_cols
    Extract column names from order_by spec

    Parameters
    ----------
    order_by : string or tuple or list of tuples
        Order by specification

    Returns
    -------
    list of column names in order_by
    '''
    cns = self.colnames
    
    # convert string to list, tuple to list
    order_by = _enlist(order_by)
    
    out = [None] * len(order_by)
    # handle order_by
    for id, x in enumerate(order_by):           
      
      # case 1: tuple
      # prototypes:
      # ('col_a', 'asc')
      # (F.col('col_a'), 'desc')
      if isinstance(x, tuple):
        assert len(x) == 2,\
          (f'Input tuple should have length 2. '
           f'Input: {x}'
           )
        assert isinstance(x[0], str),\
          (f'First element of the input tuple should be a string. '
           f'Input: {x}'
           )
        if x[1] not in ['asc', 'desc']:
          raise Exception((f'Second element of the input tuple should one '
                           f'among: ["asc", "desc"]. '
                           f'Input: {x}'
                           ))
        
        assert x[0] in cns,\
          (f'String input to order_by should be a valid column name. '
           f'Input: {x[0]}'
           )
        
        out[id] = x[0]
      
      # case 2: string
      # prototypes:
      # 'col_a'
      elif isinstance(x, str):
        assert x in cns,\
          (f'String input to order_by should be a valid column name. '
           f'Input: {x}'
           )
        out[id] = x
      
      else:
        raise Exception(("An element of 'order_by' should be a tuple or "
                         "a string"
                         ))
      
    return out
    
  def _create_windowspec(self, **kwargs):
    '''
    _create_windowspec
    Create Window object using relevant kwargs

    Parameters
    ----------
    **kwargs : 
      Supports these: by, order_by, range_between, rows_between.

    Returns
    -------
    an instance of pyspark.sql.window.WindowSpec
    
    Notes
    -----
    _create_windowspec does not validates inputs
    '''
    
    if 'by' in kwargs:
      win = Window.partitionBy(kwargs['by'])
    
    if 'order_by' in kwargs:
      if 'win' in locals():
        win = win.orderBy(kwargs['order_by'])
      else:
        win = Window.orderBy(kwargs['order_by'])
            
    if 'range_between' in kwargs:
      win = win.rangeBetween(*kwargs['range_between'])
    
    if 'rows_between' in kwargs:
      win = win.rowsBetween(*kwargs['rows_between'])
    
    return win
  
  # utils -------------------------------------------------------------------
  def add_row_number(self, order_by, name = "row_number", by = None):
    '''
    add_row_number
    Adds a column indicating row number optionally per group

    Parameters
    ----------
    order_by : order by specification
      How to order before assigning row numbers.
    name : string, optional
      Name of the new column. The default is "row_number".
    by : string or list of strings, optional
      Column names to group by. The default is None.

    Returns
    -------
    res : pyspark dataframe
    
    Examples
    --------
    (pen.ts.add_row_number('bill_length_mm')
        .show(10)
        )
    
    (pen.ts.add_row_number('bill_length_mm', by = 'species')
        .filter(F.col('row_number') <= 2)
        .show(10)
        )
    
    '''
    order_by = self._clean_order_by(order_by)
    
    if by is not None:
      by = self._clean_by(by)
      win = self._create_windowspec(by = by, order_by = order_by)
    else:
      win = self._create_windowspec(order_by = order_by)
    
    res = self.__data.withColumn(name, F.row_number().over(win))
    return res
  
  def add_group_number(self, by, name = "group_number"):
    '''
    add_group_number
    Adds group number per group

    Parameters
    ----------
    by : string or list of strings
      Column names to group by
    name : string, optional
      Name of the new column to be created. The default is "group_number".

    Returns
    -------
    res : pyspark dataframe
    
    Examples
    --------
    pen = spark.read.csv("pen.csv", header = True).drop("_c0")
    pen.show(6)
    
    (pen.ts.add_row_number('species', by = 'species')
        .filter(F.col('row_number') <= 2)
        .drop('row_number')
        .ts.add_group_number('species', name = 'gn')
        .show(10)
        )
    '''
    by = self._clean_by(by)
    win = self._create_windowspec(order_by = by)
    groups_numbered = (
      self.__data
          .select(by)
          .dropDuplicates()
          .withColumn(name, F.row_number().over(win))
          )
    
    res = self.__data.join(groups_numbered, how = "inner", on = by)       
    return res

  # basic verbs -------------------------------------------------------------
  
  def select(self, column_names, include: bool = True):
    '''
    select
    Subset some columns
    
    Parameters
    ----------
    column_names: (list of strings or a string)
        Names of the columns to be selected when 'include' is True
    
    include: (flag, default = True)
        flag to indicate whether 'column_names' should be selected or removed
    
    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    pen.ts.select('species').show(10)
    pen.ts.select(['species', 'island']).show(10)
    pen.ts.select(['species', 'island'], include = False).show(10)
    '''
    cn = self._clean_column_names(column_names)
    
    if not include:
      cn = list(set(self.colnames).difference(cn))
      if len(cn) == 0:
        raise Exception("Atleast one column should be selected in 'select'")
        
    res = self.__data.select(*cn)
    return res
  
  def arrange(self, order_by):
    '''
    arrange
    Arrange rows

    Parameters
    ----------
    order_by : string or tuple or list of tuples
        Order by specification

    Returns
    -------
    pyspark dataframe
    
    Notes
    -----
    1. 'arrange' is not memory efficient as it brings all data to a single executor
    
    Examples
    --------
    pen.ts.arrange('bill_depth_mm').show(10)
    pen.ts.arrange(['bill_depth_mm', ('bill_length_mm', 'desc')]).show(10)
    '''
    order_by = self._clean_order_by(order_by)
    warnings.warn(
        "1. 'arrange' is not memory efficient as it brings all data "
         "to a single executor"
         )
      
    res = self.__data.orderBy(order_by)
    return res
      
  def distinct(self, column_names = None, order_by = None, keep_all = False):
    '''
    distinct
    Keep only distinct combinations of columns

    Parameters
    ----------
    column_names : string or a list of strings, optional
      Column names to identify distinct rows. 
      The default is None. All columns are considered.
    order_by : order_by specification, optional
      Columns to order by to know which rows to retain. The default is None.
    keep_all : bool, optional
      Whether to keep all the columns. The default is False.

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    pen.ts.distinct('island').show(10)
    pen.ts.distinct(['species', 'island']).show(10)
    pen.ts.distinct(['species', 'island'], keep_all = True).show(10)
    '''
    if column_names is None:
      column_names = self.colnames
    
    cn = self._clean_column_names(column_names)
    
    if order_by is None:
      res = self.__data.dropDuplicates(cn)
    else:
      order_by = self._clean_order_by(order_by)
      win = self._create_windowspec(order_by = order_by)
      rank_colname = _generate_new_string(self.colnames)
      
      res = (self.__data
                 .withColumn(rank_colname, F.row_number().over(win))
                 .dropDuplicates(cn + [rank_colname])
                 .drop(rank_colname)
                 )
        
    if not keep_all:
      res = res.select(*cn)
    
    return res

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
  #
  #   '''
  #   return self.__data.toPandas()

  def rename(self, old_new_dict=None, predicate=None, func=None):
      '''
      Rename columns of the pyspark dataframe

      Parameters
      ----------
      old_new_dict: dict
          A dict with old names as keys and new names as values

      Returns
      -------
      pyspark dataframe

      Examples
      --------
        pen.ts.rename({'species': 'species_2'})
      '''
      cn = self.colnames

      if (old_new_dict is None):
          raise Exception("provide  atleast one column to change")


      assert isinstance(old_new_dict, dict), \
          "arg 'old_new_dict' should be a dict"
      assert set(cn).issuperset(old_new_dict.keys()), \
          "keys of the input dict should be existing column names"
      assert _is_string_or_string_list(list(old_new_dict.values())), \
          "values of the dict should be strings"
      assert _is_unique_list(list(old_new_dict.values())), \
          "values of the dict should be unique"


      # new names should not intersect with 'remaining' names
      remaining = set(cn).difference(old_new_dict.keys())
      assert len(remaining.intersection(old_new_dict.values())) == 0, \
          ("New intended column names (values of the dict) lead to duplicate "
           "column names"
           )

      if len(old_new_dict)>0:
          return self.__data.select([F.col(c).alias(old_new_dict.get(c, c)) for c in self.colnames])
      else:
          return self.__data



  def relocate(self, column_names, before = None, after = None):
      '''
      relocate the columns of the pyspark dataframe

      Parameters
      ----------
      column_names : string or a list of strings
          column names to be moved
      before : string, optional
          column before which the column are to be moved. The default is None.
      after : TYPE, optional
          column after which the column are to be moved. The default is None.


      Returns
      -------
      pyspark dataframe

      Notes
      -----
      Only one among 'before' and 'after' can be not None. When both are None,
      the columns are added to the begining of the dataframe (leftmost)

      Examples
      --------


        # move "island" and "species" columns to the left of the dataframe
        pen.ts.relocate(["island", "species"])

        # move "sex" and "year" columns to the left of "island" column
        pen.ts.relocate(["sex", "year"], before = "island")

         # move "island" and "species" columns to the right of "year" column
        pen.ts.relocate(["island", "species"], after = "year")
      '''
      column_names = self._clean_column_names(column_names)
      cn = self.colnames

      cn = self.colnames
      col_not_relocate = [i for i in cn if i not in column_names]

      assert not ((before is not None) and (after is not None)), \
          "Atleast one arg among 'before' and 'after' should be None"

      if after is not None:
          assert isinstance(after, str), \
              "arg 'after' should be a string"
          assert after in cn, \
              "arg 'after' should be a exisiting column name"
          assert not (after in column_names), \
              "arg 'after' should be an element of 'column_names'"

      if before is not None:
          assert isinstance(before, str), \
              "arg 'before' should be a string"
          assert before in cn, \
              "arg 'before' should be a exisiting column name"
          assert not (before in column_names), \
              "arg 'before' should be an element of 'column_names'"

      # case 1: relocate to start when both before and after are None
      if (before is None) and (after is None):
          col_sequence = column_names + list(col_not_relocate)
      elif (before is not None):
          # case 2: before is not None
          col_sequence = col_not_relocate[0:col_not_relocate.index(before)] + \
                          column_names + \
                          col_not_relocate[col_not_relocate.index(before):]

      else:
          # case 3: after is not None
          after_index = col_not_relocate.index(after)
          col_sequence = col_not_relocate[0:after_index+1] + \
                         column_names + \
                         col_not_relocate[after_index+1:]

      res = self.__data.select(col_sequence)
      return res

  def mutate(self, dictionary, window_spec = None, **kwargs):
    '''
    mutate
    Create new column or modify existing columns

    Parameters
    ----------
    dictionary : dict
      key should be new/existing column name. 
      Value should is pyspark expression that should evaluate to a column
    window_spec : pyspark.sql.window.WindowSpec, optional
      The default is None.
    **kwargs : Supports these: by, order_by, range_between, rows_between.

    Returns
    -------
    pyspark dataframe
      
    Examples
    --------
    (pen.ts.mutate({'bl_+_1': F.col('bill_length_mm') + 1,
                     'bl_+_1_by_2': F.col('bl_+_1') / 2})
        .show(10)
        )
    
    # grouped and order mutate operation
    (pen.ts.add_row_number(order_by = 'bill_depth_mm')
        .ts.mutate({'cumsum_bl': F.sum('bill_length_mm')},
                   by = 'species',
                   order_by = ['bill_depth_mm', 'row_number'],
                   range_between = (-float('inf'), 0)
                   )
        .ts.select(['bill_length_mm',
                    'species',
                    'bill_depth_mm',
                    'cumsum_bl'
                    ])
        .show(10)
        )
    
    '''
    res = self.__data
    
    with_win = False
    # windowspec gets first preference
    if window_spec is not None:
      with_win = True
      assert isinstance(window_spec, pyspark.sql.window.WindowSpec),\
        ("'window_spec' should be an instance of"
         "'pyspark.sql.window.WindowSpec' class"
         )
      if len(kwargs) >= 1:
        print("'window_spec' takes precedence over other kwargs"
              " in mutate"
              )
    else:
      # create windowspec if required
      if len(kwargs) >= 1:
        win = self._create_windowspec(**kwargs)
        with_win = True
    
    # create one column at a time
    for key, value in dictionary.items():
      if not with_win:
        res = res.withColumn(key, value)
      else:
        res = res.withColumn(key, (value).over(win))
            
    return res
  
  def summarise(self, dictionary, by = None):
    '''
    summarise
    Create aggreagate columns

    Parameters
    ----------
    dictionary : dict
      key should be column name. 
      Value should is pyspark expression that should produce a single
      aggregation value
    by : string or list of strings
      Column names to group by

    Returns
    -------
    pyspark dataframe
      
    Examples
    --------
    # ungrouped summarise
    (pen.ts.summarise({'mean_bl': F.mean(F.col('bill_length_mm')),
                       'count_species': F.count(F.col('species'))
                      }
                     )
        .show(10)
        )
    
    # grouped summarise
    (pen.ts.summarise({'mean_bl': F.mean(F.col('bill_length_mm')),
                       'count_species': F.count(F.col('species'))
                      },
                      by = 'island'
                     )
        .show(10)
        )
    '''
    
    agg_list = [tup[1].alias(tup[0]) for tup in dictionary.items()]
    
    if by is None:
      res = self.__data.agg(*agg_list)
    else:
      by = self._clean_by(by)
      res = self.__data.groupBy(*by).agg(*agg_list)
    
    return res
  
  summarize = summarise
    
  # joins --------------------------------------------------------------------
  def join(self, pyspark_df, on = None, sql_on = None):
    '''
    TODO -- srikanth

    Parameters
    ----------
    pyspark_df : TYPE
      DESCRIPTION.
    on : TYPE, optional
      DESCRIPTION. The default is None.
    sql_on : TYPE, optional
      DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
      DESCRIPTION.

    '''
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
      "'pyspark_df' should be a pyspark dataframe"
    
    assert ((on is None) + (sql_on is None) == 1),\
      "Exactly one among 'on', 'sql_on' should be specified"
    
    if on is not None:
      assert _is_string_or_string_list(on),\
          ("'on' should be a string or a list of strings of common "
           "column names"
           )
        
    if sql_on is not None:
      assert isinstance(sql_on, str)
    
    LHS = self.__data
    RHS = pyspark_df
    
    if on is not None:
      res = LHS.join(RHS, on = on, how = "inner")
    else:
      res = LHS.join(RHS, on = eval(sql_on), how = "inner")
        
    return res
  
  # count methods ------------------------------------------------------------
  def count(self, column_names, name = 'n'):
    '''
    TODO -- srikanth

    Parameters
    ----------
    column_names : TYPE
      DESCRIPTION.
    name : TYPE, optional
      DESCRIPTION. The default is 'n'.

    Returns
    -------
    res : TYPE
      DESCRIPTION.

    '''
    cn = self._clean_column_names(column_names)
    
    assert isinstance(name, str),\
        "'name' should be a string"
    assert name not in column_names,\
        "'name' should not be a element of 'column_names'"
    
    res = (self.__data
               .select(*cn)
               .withColumn(name, F.lit(1))
               .groupby(*cn)
               .agg(F.sum(F.col(name)).alias(name))
               )
    return res
