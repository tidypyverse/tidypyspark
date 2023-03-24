# ----------------------------------------------------------------------------
# This file is a part of tidypyspark python package
# Find the dev version here: https://github.com/talegari/tidypyspark
# ---------------------------------------------------------------------
import warnings
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import re
import sys
import json
import numpy as np
from collections_extended import setlist
from collections import Counter
from tidypyspark.accessor_class import register_dataframe_accessor
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
                              _nested_is_unique,
                              _get_compatible_datatypes_of_python_and_spark
                            )

# tidypyspark ----
@register_dataframe_accessor('ts')
class acc_on_pyspark():
  '''
  
  '''
  def __init__(self, data):
    
    # assign __data attribute
    colnames = list(data.columns)
    assert all([_is_valid_colname(x) for x in colnames]),\
      "column names should not start with underscore"
    assert _is_unique_list(colnames),\
      "column names should be unique"
    self.__data = data
  
  # attributes -------------------------------------------------------------
  @property
  def nrow(self):
    '''
    nrow
    get munber of rows
    
    Notes
    -----
    Just a placeholder to inform user to use count() method
    
    Returns
    -------
    None
    '''
    print("Please run: `df.ts.count()` to get the number of rows")
    return None
  
  @property
  def ncol(self):
    '''
    ncol
    get number of columns

    Returns
    -------
    (Integer) Number of columns
    '''
    return len(self.__data.columns)
  
  @property
  def colnames(self):
    '''
    dim
    get column names

    Returns
    -------
    list with column names as strings
    '''
    return list(self.__data.columns)
  
  @property
  def shape(self):
    '''
    shape
    get shape

    Returns
    -------
    tuple of number of rows and number of columns
    '''
    return (self.nrow, self.ncol)
  
  @property
  def dim(self):
    '''
    dim
    get shape

    Returns
    -------
    tuple of number of rows and number of columns
    '''
    return (self.nrow, self.ncol)
  
  @property
  def types(self):
    '''
    types
    identify column types as strings

    Returns
    -------
    dict with column names as keys, types as a values
    type is a string.
    '''
    names = self.__data.columns
    types = [x.dataType.typeName() for x in self.__data.schema.fields]
    res = dict(zip(names, types))
    return res

  # cleaners --------------------------------------------------------------
  def _validate_by(self, by):
    '''
    _validate_by
    validates 'by' and returns a list of column names
    
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

  def _validate_order_by(self, order_by):
    '''
    _validate_order_by
    validates and returns a list of 'column' objects
    
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
        allowed = ['asc', 'desc', 'asc_null_first', 'desc_nulls_first']
        if x[1] not in allowed:
          raise Exception((f'Second element of the input tuple should '
                           f'one among: {allowed}. '
                           f'Input: {x}'
                           ))
        
        assert x[0] in cns,\
          (f"String input to 'order_by' should be a valid column name. "
           f"Input: {x[0]}"
           )
          
        if x[1] == 'asc':
          order_by[id] = F.col(x[0]).asc_nulls_last()
        elif x[1] == 'asc_nulls_first':
          order_by[id] = F.col(x[0]).asc_nulls_first()
        elif x[1] == 'desc':
          order_by[id] = F.col(x[0]).desc_nulls_last()
        else:
          order_by[id] = F.col(x[0]).desc_nulls_first()
      
      # case 2: string
      # prototypes:
      # 'col_a'
      elif isinstance(x, str):
        assert x in cns,\
          (f"String input to 'order_by' should be a valid column name. "
           f'Input: {x}'
           )
        order_by[id] = F.col(x).asc_nulls_last()
      
      else:
        raise Exception(("An element of 'order_by' should be a tuple or "
                         "a string"))
      
    return order_by
      
  def _validate_column_names(self, column_names):
    '''
    _validate_column_names
    validates and returns column names
    
    Parameters
    ----------
    column_names : string or list of strings
      columns names to be validated

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
    Extract column names from order_by specification

    Parameters
    ----------
    order_by : string or tuple or list of tuples
        Order by specification
        ex: ["col_a", ("col_b", "desc")]

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
        allowed = ['asc', 'desc', 'asc_null_first', 'desc_nulls_first']
        if x[1] not in allowed:
          raise Exception((f'Second element of the input tuple should one '
                           f'among: ["asc", "desc"]. '
                           f'Input: {x}'
                           ))
        
        assert x[0] in cns,\
          (f"String input to 'order_by' should be a valid column name. "
           f"Input: {x[0]}"
           )
        
        out[id] = x[0]
      
      # case 2: string
      # prototypes:
      # 'col_a'
      elif isinstance(x, str):
        assert x in cns,\
          (f"String input to 'order_by' should be a valid column name. "
           f"Input: {x}"
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
    
    valid_kwarg_names = ["by", "order_by", "range_between", "rows_between"]
    assert set(kwargs.keys()).issubset(valid_kwarg_names),\
      f"Input kwarg should have one of these names: {valid_kwarg_names}"
    
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

  def glimpse(self, n_rows=100, n_columns = 100):
    '''
    glimpse

    This function prints the first 100(default) rows of the dataset, along with
    the number of rows and columns in the dataset and the data types of each
    column. It also displays the values of each column, limited to the first
    100 (default) values. If the number of rows exceeds 100, then the values are
    truncated accordingly.

    Parameters
    ----------
    n_columns : maximum number of columns to be handled
    n_rows: maximum number of rows to show

    Returns
    -------
    None

    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.glimpse()
    '''
    from shutil import get_terminal_size
    w, h = get_terminal_size()

    # get row and column count
    ncol = len(self.__data.columns)

    res = [f'Columns: {ncol}']

    names = []
    n_ljust = 0
    dtypes = []
    t_ljust = 0

    # Input validation
    assert isinstance(n_rows, int) and n_rows > 0,\
      "n_rows must be a positive integer"
    assert isinstance(n_columns, int) and n_columns > 0,\
      "n_columns must be a positive integer"

    #get dtypes for all columns
    data_temp = self.__data.limit(n_rows).toPandas()
    all_dtypes = self.types

    for acol in self.colnames[0:n_columns]:
      names.append(acol)
      n_ljust = max(n_ljust, len(names[-1]))
      dtypes.append(f'<{all_dtypes[acol]}>')
      t_ljust = max(t_ljust, len(dtypes[-1]))

    #get values for float and non float columns
    float_vals_precision = 2
    for name, dtype in zip(names, dtypes):
      vals = data_temp.loc[0:n_rows, name]
      if dtype[1:6] == "float":
          vals = vals.round(float_vals_precision)
      val_str = ", ".join(list(map(str, vals)))

      if len(val_str) > w-2-n_ljust-t_ljust:
          val_str = val_str[0:(w-2-n_ljust-t_ljust)-3] + "..."
      res_str = f'{name.ljust(n_ljust)} {dtype.ljust(t_ljust)} {val_str}'
      res.append(res_str)

    # print additional columnnames/count in case it is more then n_rows
    if ncol > n_columns:
      footer = (f'\nmore columns: '
                f'{", ".join(self.colnames[n_columns:(n_columns+50)])}')
      if ncol >= n_columns+50:
        footer += "..."
        res.append(footer)
    print("\n".join(res))
    return None
  

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
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.add_row_number('bill_length_mm').show(10)
    >>> (pen.ts.add_row_number('bill_length_mm', by='species')
    >>>     .filter(F.col('row_number') <= 2)
    >>>     .show(10))
    '''
    order_by = self._validate_order_by(order_by)
    assert name not in self.colnames,\
      f"{name} should not be an existing column name"
    if by is not None:
      by = self._validate_by(by)
      win = self._create_windowspec(by = by, order_by = order_by)
    else:
      win = self._create_windowspec(order_by = order_by)
    
    res = self.__data.withColumn(name, F.row_number().over(win))
    return res
  
  # alias
  rowid_to_column = add_row_number
  
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
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> (pen.ts.add_row_number('species', by = 'species')
    >>>     .filter(F.col('row_number') <= 2)
    >>>     .drop('row_number')
    >>>     .ts.add_group_number('species', name = 'gn')
    >>>     .show(10)
    >>>     )
    '''
    assert name not in self.colnames,\
      f"{name} should not be an existing column name"
    by = self._validate_by(by)
    win = self._create_windowspec(order_by = by)
    groups_numbered = (self.__data
                           .select(by)
                           .dropDuplicates()
                           .withColumn(name, F.row_number().over(win))
                           )
    res = self.__data.join(groups_numbered, how = "inner", on = by)       
    return res
  
  def to_list(self, column_name):
    '''
    to_list
    collect a column as python list

    Parameters
    ----------
    column_name : str
      Name of the column to be collected.

    Returns
    -------
    list
    '''
    assert isinstance(column_name, str)
    column_name = self._validate_column_names(column_name)[0]
    res = (self.__data
               .select(column_name)
               .rdd.map(lambda x: x[0])
               .collect()
               )
    return res
  
  def pull(self, column_name):
    '''
    pull
    collect a column as pandas series

    Parameters
    ----------
    column_name : str
      Name of the column to be collected.

    Returns
    -------
    pandas series
    '''
    assert isinstance(column_name, str)
    column_name = self._validate_column_names(column_name)[0]
    res = (self.__data
               .select(column_name)
               .rdd.map(lambda x: x[0])
               .collect()
               )
    import pandas as pd
    res = pd.Series(res, name = column_name[0])
      
    return res
  
  # alias for pull
  to_series = pull
  
  def to_dict(self):
    '''
    to_dict
    collect as a dict where keys are column names and
    values are lists

    Returns
    -------
    dict
    
    Notes
    -----
    Each column is pulled separately to reduce the load on the executor.
    '''
    res_dict = {}
    for acolname in self.colnames:
      res_dict[acolname] = (self.__data
                                .select(acolname)
                                .rdd.map(lambda x: x[0])
                                .collect()
                                )
    return res_dict
  
  def to_pandas(self):
    '''
    to_pandas
    collect as a pandas dataframe

    Returns
    -------
    pandas dataframe
    '''
    return self.__data.toPandas()
  
  # alias: to_pandas
  collect = to_pandas

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
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.select('species').show(10)
    >>> pen.ts.select(['species', 'island']).show(10)
    >>> pen.ts.select(['species', 'island'], include = False).show(10)
    '''
    cn = self._validate_column_names(column_names)
    
    if not include:
      cn = list(set(self.colnames).difference(cn))
      if len(cn) == 0:
        raise Exception("Atleast one column should be selected")
        
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
    1. 'arrange' is not memory efficient as it brings all data
       to a single executor.
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.arrange('bill_depth_mm').show(10)
    >>> pen.ts.arrange(['bill_depth_mm', ('bill_length_mm', 'desc')]).show(10)
    '''
    order_by = self._validate_order_by(order_by)
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
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.distinct('island').show(10)
    >>> pen.ts.distinct(['species', 'island']).show(10)
    >>> pen.ts.distinct(['species', 'island'], keep_all = True).show(10)
    '''
    if column_names is None:
      column_names = self.colnames
    
    cn = self._validate_column_names(column_names)
    
    if order_by is None:
      res = self.__data.dropDuplicates(cn)
    else:
      order_by = self._validate_order_by(order_by)
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

  def rename(self, old_new_dict):
    '''
    rename
    Rename columns

    Parameters
    ----------
    old_new_dict: dict
      A dict with old names as keys and new names as values

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.rename({'species': 'species_2', "year":"year_2})
    '''
    cn = self.colnames

    assert isinstance(old_new_dict, dict), \
      "arg 'old_new_dict' should be a dict"
    assert len(old_new_dict) > 0, \
      "There should alteast one element in old_new_dict"
    
    new_column_names = list(old_new_dict.values())
    
    assert set(cn).issuperset(old_new_dict.keys()), \
      "Keys of the input dict should be existing column names"
    assert _is_unique_list(new_column_names), \
      "Values of the dict should be unique"
    assert all([_is_valid_colname(x) for x in new_column_names]),\
      "Atleast one value of the dict is not a valid column name"
    
    # new names should not intersect with 'remaining' names
    remaining = set(cn).difference(old_new_dict.keys())
    assert len(remaining.intersection(old_new_dict.values())) == 0, \
      ("New intended column names (values of the dict) should not lead to "     
       "duplicate column names"
       )
    select_col_with_alias = ([F.col(c).alias(old_new_dict.get(c, c))
                             for c in self.colnames])
    res = self.__data.select(select_col_with_alias)
    return res

  def relocate(self, column_names, before = None, after = None):
    '''
    relocate
    Relocate the columns

    Parameters
    ----------
    column_names : string or a list of strings
      column names to be moved
    before : string, optional
      column before which the column_names are to be moved.
      The default is None.
    after : string, optional
      column after which the column_names are to be moved.
      The default is None.

    Returns
    -------
    pyspark dataframe

    Notes
    -----
    Only one among 'before' and 'after' can be not None. When both are None,
    the columns are added to the beginning of the dataframe (leftmost)

    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> # move "island" and "species" columns to the left of the dataframe
    >>> pen.ts.relocate(["island", "species"])
    >>> # move "sex" and "year" columns to the left of "island" column
    >>> pen.ts.relocate(["sex", "year"], before = "island")
    >>> # move "island" and "species" columns to the right of "year" column
    >>> pen.ts.relocate(["island", "species"], after = "year")
    '''

    column_names = self._validate_column_names(column_names)
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
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> (pen.ts.mutate({'bl_+_1': F.col('bill_length_mm') + 1,
    >>>                  'bl_+_1_by_2': F.col('bl_+_1') / 2})
    >>>     .show(10)
    >>>     )
    >>> # grouped and order mutate operation
    >>> (pen.ts.add_row_number(order_by = 'bill_depth_mm')
    >>>     .ts.mutate({'cumsum_bl': F.sum('bill_length_mm')},
    >>>                by = 'species',
    >>>                order_by = ['bill_depth_mm', 'row_number'],
    >>>                range_between = (-float('inf'), 0)
    >>>                )
    >>>     .ts.select(['bill_length_mm',
    >>>                 'species',
    >>>                 'bill_depth_mm',
    >>>                 'cumsum_bl'
    >>>                 ])
    >>>     .show(10)
    >>>     )
    '''
    assert all([_is_valid_colname(x) for x in dictionary.keys()]),\
      "Atleast one key of the dictionary is not a valid column name"
    res = self.__data
  
    with_win = False
    # windowspec gets first preference
    if window_spec is not None:
      with_win = True
      assert isinstance(window_spec, pyspark.sql.window.WindowSpec),\
        ("'window_spec' should be an instance of "
         "'pyspark.sql.window.WindowSpec' class"
         )
      if len(kwargs) >= 1:
        print("'window_spec' takes precedence over other kwargs"
              " in mutate"
              )
    else:
      # create windowspec if required
      if len(kwargs) >= 1:
        for akwarg_name, akwarg_value in kwargs.items():
          if akwarg_name == "by":
            kwargs[akwarg_name] = self._validate_by(kwargs[akwarg_name])
          if akwarg_name == "order_by":
            kwargs[akwarg_name] = self._validate_order_by(kwargs[akwarg_name])
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
    Create aggregate columns

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
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> # ungrouped summarise
    >>> (pen.ts.summarise({'mean_bl': F.mean(F.col('bill_length_mm')),
    >>>                    'count_species': F.count(F.col('species'))
    >>>                   }
    >>>                  )
    >>>     .show(10)
    >>>     )
    >>> 
    >>> # grouped summarise
    >>> (pen.ts.summarise({'mean_bl': F.mean(F.col('bill_length_mm')),
    >>>                    'count_species': F.count(F.col('species'))
    >>>                   },
    >>>                   by = 'island'
    >>>                  )
    >>>     .show(10)
    >>>     )
    '''
    assert all([_is_valid_colname(x) for x in dictionary.keys()]),\
      "Atleast one key of the dictionary is not a valid column name"
    agg_list = [tup[1].alias(tup[0]) for tup in dictionary.items()]
    
    if by is None:
      res = self.__data.agg(*agg_list)
    else:
      by = self._validate_by(by)
      res = self.__data.groupBy(*by).agg(*agg_list)
    
    return res
  
  summarize = summarise
  
  def filter(self, condition):
    '''
    filter
    subset rows using some condition

    Parameters
    ----------
    condition : pyspark column or string
      Column of types.BooleanType or a string of SQL expression.

    Returns
    -------
    pyspark dataframe
    '''
    return self.__data.filter(condition)
    
  # join methods ------------------------------------------------------------
  def _validate_join(self, pyspark_df, on, on_x, on_y , sql_on, suffix, how):
      
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
      "'pyspark_df' should be a pyspark dataframe"
    
    assert isinstance(how, str),\
      "arg 'how' should be a string"
    valid_joins = ['inner', 'left', 'right', 'full', 'semi', 'anti']
    assert how in valid_joins,\
      f"arg 'how' should be one among: {valid_joins}"
    
    cn_y = pyspark_df.columns
    assert _is_unique_list(cn_y),\
      "column names of 'pyspark_df' should be unique"
  
    assert ((on is not None) 
            + ((on_x is not None) and (on_y is not None)) 
            + (sql_on is not None) == 1),\
      "Exactly one among 'on', 'sql_on', 'on_x and on_y' should be specified"
  
    if on is not None:
      on = self._validate_column_names(on)
      assert set(on).issubset(cn_y),\
        "arg 'on' should be a subset of column names of y"
    elif sql_on is not None:
      assert isinstance(sql_on, str),\
        "arg 'sql_on' should be a string"
    else:
      assert on_x is not None and on_y is not None,\
        ("When arg 'on' is None, " 
          "both args 'on_x' and 'on_y' should be present"
        )
        
      on_x = _enlist(on_x)
      on_y = _enlist(on_y)
      
      on_x = self._validate_column_names(on_x)
      
      assert _is_string_or_string_list(on_y),\
        "arg 'on_y' should be a string or a list of strings"
      
      assert _is_unique_list(on_y),\
        "arg 'on_y' should not have duplicates"    
      assert set(on_y).issubset(cn_y),\
        "arg 'on_y' should be a subset of column names of y"
      assert len(on_x) == len(on_y),\
        "Lengths of arg 'on_x' and arg 'on_y' should match"
      
    assert _is_string_or_string_list(suffix) and len(suffix) == 2,\
      "arg 'suffix' should be a list of strings of length 2"
    
    assert suffix[0] != suffix[1],\
      "left and right suffix should be different."
    
    return None

  def _execute_on_command_for_join(self, on, suffix, how, LHS, RHS):
          
    on = _enlist(on)
    cols_LHS = LHS.columns
    cols_RHS = RHS.columns

    # Create a dictionary with old column names as keys and values 
    # as old column names + suffix
    new_cols_LHS = list(setlist(cols_LHS) - setlist(on))
    old_new_dict_LHS = {col: col + suffix[0] for col in new_cols_LHS}      

    # Create a dictionary with old column names as keys and values
    # as old column names + suffix
    new_cols_RHS = list(setlist(cols_RHS) - setlist(on))
    old_new_dict_RHS = {col: col + suffix[1] for col in new_cols_RHS}

    int_new_cols = (set(old_new_dict_LHS.values())
                    .intersection(old_new_dict_RHS.values())
                    )
    assert len(int_new_cols) == 0,\
      "Resulting column names should be unique after joining the dataframes"
    
    # Create a list of columns with alias for LHS df in pyspark convention.
    select_col_with_alias_LHS = (
      [F.col(c).alias(old_new_dict_LHS.get(c, c))
       for c in cols_LHS
       ])
    # Create the new LHS df with the new column names.
    new_LHS = LHS.select(select_col_with_alias_LHS)
    
    # Create a list of columns with alias for RHS df in pyspark convention.
    select_col_with_alias_RHS = (
      [F.col(c).alias(old_new_dict_RHS.get(c, c))
       for c in cols_RHS
       ])
    # Create the new RHS df with the new column names.
    new_RHS = RHS.select(select_col_with_alias_RHS)

    # Join the dataframes
    res = new_LHS.join(new_RHS, on = on, how = how)

    # Get the count of columns in the original joined dataframe
    count_cols = Counter(cols_LHS + cols_RHS)
    
    renamed_col_dict = {} # Dictionary to store the renamed columns
    for col in res.columns: 
      # use a loop to check each suffix in the list
      for s in suffix: 
        # check if the column name ends with the suffix
        if len(s) > 0 and col.endswith(s): 
          # remove the suffix from the column name
          new_col = col[:-len(s)] 
          # Check if the new column name is not a duplicate and
          # is not in the list of columns to be joined
          if count_cols[new_col] < 2 and new_col not in on:      
            renamed_col_dict[col] = new_col 

    # Rename the columns in the dataframe
    if len(renamed_col_dict) > 0:
      res = res.ts.rename(renamed_col_dict)

    return res
  
  def _execute_sql_on_command_for_join(self, sql_on, suffix, how, LHS, RHS):

    cols_LHS = LHS.columns
    cols_RHS = RHS.columns

    # Create a dictionary with old column names as keys and values
    # as old column names + suffix
    old_new_dict_LHS = {col: col + suffix[0] for col in cols_LHS}
    old_new_dict_RHS = {col: col + suffix[1] for col in cols_RHS}

    int_new_cols = (set(old_new_dict_LHS.values())
                    .intersection(old_new_dict_RHS.values())
                    )
    
    assert len(int_new_cols) == 0,\
      "Resulting column names should be unique after joining the dataframes"

    # Get the column names from the sql_on command.
    # e.g Format - {'LHS': ['dept', 'id'], 'RHS': ['dept', 'id', 'age']}
    column_names_tuple_list = self._extract_cols_from_sql_on_command(sql_on)
    
    # Get the sql_on statement with suffix.
    # e.g Format - "LHS.dept = RHS.dept_y and LHS.id = RHS.id_y"
    sql_on = self._get_sql_on_statement_with_suffix(sql_on,
                                                    suffix,
                                                    column_names_tuple_list
                                                    )

    # Rename the columns of the LHS and RHS dataframes.
    LHS = LHS.ts.rename(old_new_dict_LHS) 
    RHS = RHS.ts.rename(old_new_dict_RHS)

    # Join the dataframes
    res = LHS.join(RHS, on = eval(sql_on), how = how)

    # Get the count of columns in the original joined dataframe
    res = self._get_spark_df_by_removing_suffix(suffix,
                                                cols_LHS,
                                                cols_RHS,
                                                res
                                                )

    return res
  
  def _execute_on_x_on_y_command_for_join(self,
                                          on_x,
                                          on_y,
                                          suffix,
                                          how,
                                          LHS,
                                          RHS):

    on_x = _enlist(on_x)
    on_y = _enlist(on_y)

    # Degenrate case when on_x and on_y are equal. 
    if on_x == on_y:
      return self._execute_on_command_for_join(on_x, suffix, how, LHS, RHS)

    # Generate the sql_on command from on_x and on_y
    # e.g Format - "(LHS.dept == RHS.dept) & (LHS.id == RHS.id)"
    sql_on = ""
    # Iterate over the on_x and on_y lists to create the sql_on command.
    for i in range(len(on_x)):
        if i == 0:
            sql_on += "(LHS." + on_x[i] + " == RHS." + on_y[i] + ")"
        else:
            sql_on += " & (LHS." + on_x[i] + " == RHS." + on_y[i] + ")"

    # Execute the sql_on command to return the joined dataframe.
    res = self._execute_sql_on_command_for_join(sql_on, suffix, how, LHS, RHS)

    return res

  def _get_sql_on_statement_with_suffix(self,
                                        sql_on,
                                        suffix,
                                        column_names_tuple_list):
      
    # Create a list of column names with suffix for LHS and RHS.
    # e.g Format - ['LHS.dept', 'LHS.id', 'RHS.dept', 'RHS.id', 'RHS.age']
    sql_on_LHS_cols = ['LHS.' + col for col in column_names_tuple_list['LHS']]
    sql_on_RHS_cols = ['RHS.' + col for col in column_names_tuple_list['RHS']]

    # Create a dictionary with old column names as keys and values as 
    # old column names + suffix
    # e.g Format - {'LHS.dept': 'new_LHS.dept_suffix[0]', 
    #               'RHS.dept': 'new_RHS.dept_suffix[1]',
    #               'RHS.age': 'new_RHS.age_suffix[1]'}
    sql_on_LHS_cols_dict = {col: col + suffix[0] for col in sql_on_LHS_cols}
    sql_on_RHS_cols_dict = {col: col + suffix[1] for col in sql_on_RHS_cols}
    # Merge the two dictionaries
    sql_on_cols_dict = {**sql_on_LHS_cols_dict, **sql_on_RHS_cols_dict}
    
    # Replace the column names in the sql_on command with the new 
    # column names in the sql_on_cols_dict
    for key, value in sql_on_cols_dict.items():
      sql_on = sql_on.replace(key, value)
    
    return sql_on
  
  def _extract_cols_from_sql_on_command(self, sql_on):

    # Set regex pattern.
    # e.g. 'LHS.dept == RHS.dept' will be converted to
    # [('LHS', 'dept'), ('RHS', 'dept')]
    pattern = '([a-zA-Z0-9]+)\.([a-zA-Z0-9]+)'

    # Get all column names with their table names
    # Format - [('LH', 'dept'), ('LHS', 'dept'), ('RHS', 'age')]
    column_names_tuple_list = re.findall(pattern, sql_on)

    # Filtering tuples having only LHS or RHS as the first element of the tuple
    filtered_tuples = ([(key, value) 
                        for (key, value) in column_names_tuple_list 
                        if key=='LHS' or key=='RHS']
                      )

    # Generates a dictionary with LHS and RHS as keys and column names as values.
    # e.g. {'RHS': ['id2', 'dept', 'age'], 'LHS': ['dept']}
    dict_from_tuple = {key:[] for (key, _) in filtered_tuples}
    for tpl in filtered_tuples:
        dict_from_tuple[tpl[0]].append(tpl[1])

    return dict_from_tuple
  
  def _get_spark_df_by_removing_suffix(self, suffix, cols_LHS, cols_RHS, res):
      
    # Get the frequency count of columns in the original joined dataframe.
    count_cols = Counter(cols_LHS + cols_RHS)
  
    renamed_col_dict = {}
    for col in res.columns: 
      # use a loop to check each suffix in the list
      for s in suffix:
        # check if the column name ends with the suffix
        if len(s) > 0 and col.endswith(s):  
          # remove the suffix from the column name
          new_col = col[:-len(s)] 
          # Check if the new column name is not a duplicate
          if count_cols[new_col] < 2:
            renamed_col_dict[col] = new_col

    # Rename the columns in the dataframe
    if len(renamed_col_dict) > 0:
      res = res.ts.rename(renamed_col_dict)
      
    return res
  
  def _execute_cross_join(self, pyspark_df, suffix):
      
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
    "'pyspark_df' should be a pyspark dataframe"
  
    assert isinstance(suffix, list),\
    "arg 'suffix' should be a list"
        
    assert len(suffix) == 2,\
    "arg 'suffix' should be a list of length 2"
      
    assert isinstance(suffix[0], str) and isinstance(suffix[1], str),\
    "arg 'suffix' should be a list of two strings"
  
    assert suffix[0] != suffix[1],\
    "left and right suffix should be different."
  
    LHS = self.__data
    RHS = pyspark_df

    # Get the column names of the LHS and RHS dataframes
    cols_LHS = LHS.columns
    cols_RHS = RHS.columns

    # Create a dictionary with old column names as keys and values
    # as old column names + suffix
    old_new_dict_LHS = {col: col + suffix[0] for col in cols_LHS}
    old_new_dict_RHS = {col: col + suffix[1] for col in cols_RHS}
    
    int_new_cols = (set(old_new_dict_LHS.values())
                    .intersection(old_new_dict_RHS.values())
                    )
    assert len(int_new_cols) == 0,\
      "Resulting column names should be unique after joining the dataframes" 
    
    # Rename the columns in the LHS and RHS dataframes
    LHS = LHS.ts.rename(old_new_dict_LHS)
    RHS = RHS.ts.rename(old_new_dict_RHS)
  
    # Perform cross join.
    res = LHS.crossJoin(RHS)

    # Remove the unnecessary suffix(es) from the column names
    res = self._get_spark_df_by_removing_suffix(suffix,
                                                cols_LHS,
                                                cols_RHS,
                                                res
                                                )

    return res
  
  def join(self, 
           pyspark_df, 
           on = None, 
           on_x = None, 
           on_y = None, 
           sql_on = None, 
           suffix = ["", "_y"], 
           how = 'inner'
           ): 
    '''
    Joins columns of y to self
    
    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join
    how: string
      Type of join to be performed. Default is 'inner'.
      Supports: 'inner', 'left', 'right', 'outer', 'full', 'cross',
                'semi', 'anti'
      
    Returns
    -------
    pyspark dataframe

    Examples
    --------
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.join(df2, on = ['id', 'dept'], how = 'inner).show()
    +---+----+----+---+
    | id|dept|name|age|
    +---+----+----+---+
    |  1| SDE|jack| 20|
    |  2|  PM|jack| 30|
    |  2| SDE|jack| 10|
    +---+----+----+---+

    >>> (df1.ts.join(df2, 
    >>>        sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & 
    >>>        (RHS.age < 30)', how = 'inner).show())
    +---+----+----+----+------+---+
    | id|name|dept|id_y|dept_y|age|
    +---+----+----+----+------+---+
    |  1|jack| SDE|   1|   SDE| 20|
    |  2|jack| SDE|   2|   SDE| 10|
    +---+----+----+----+------+---+

    '''
 
    # Special case of cross join.
    if how == 'cross':
      return self._execute_cross_join(pyspark_df = pyspark_df, suffix = suffix)

    # Validate the input arguments for all joins except cross join.
    self._validate_join(pyspark_df, on, on_x, on_y, sql_on, suffix, how)
   
    LHS = self.__data
    RHS = pyspark_df
  
    if on is not None:
      res = self._execute_on_command_for_join(on, suffix, how, LHS, RHS)
    elif sql_on is not None:
      res = self._execute_sql_on_command_for_join(sql_on,
                                                  suffix,
                                                  how,
                                                  LHS, RHS
                                                  )
    else:
      res = self._execute_on_x_on_y_command_for_join(on_x, on_y,
                                                     suffix,
                                                     how,
                                                     LHS, RHS
                                                     )

    return res

  def left_join(self, 
                pyspark_df, 
                on = None, 
                on_x = None, 
                on_y = None , 
                sql_on = None, 
                suffix = ["", "_y"]
                ):
    '''
    Joins columns of the given pyspark_df to self by matching rows.
    Includes all keys from self.

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join
      
    Returns
    -------
    pyspark dataframe

    Examples
    --------
    # Create the DataFrames
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.left_join(df2, on = ["id", "dept"]).show()
    +---+-----+----+---+---+
    | id|dept|  name| age|
    +---+----+------+----+
    |  1|  DS|jordan|null|
    |  2|  DS|  jack|null|
    |  1| SDE|  jack|  10|
    |  2| SDE|  jack|  20|
    |  1|  PM|  jack|null|
    |  2|  PM|  jack|  30|
    +---+----+------+----+

    >>> (df1.ts.left_join(df2, sql_on = '(LHS.id == RHS.id) & 
    >>>                   (LHS.dept == RHS.dept) & (RHS.age < 30)').show())
    +---+------+----+----+------+----+
    | id|  name|dept|id_y|dept_y| age|
    +---+------+----+----+------+----+
    |  1|jordan|  DS|null|  null|null|
    |  2|  jack|  DS|null|  null|null|
    |  1|  jack| SDE|   1|   SDE|  20|
    |  2|  jack| SDE|   2|   SDE|  10|
    |  1|  jack|  PM|null|  null|null|
    |  2|  jack|  PM|null|  null|null|
    +---+------+----+----+------+----+

    >>> df1.ts.left_join(df2, on_x = ['name'], on_y = ['dept']).show()
    +---+------+----+----+------+----+
    | id|  name|dept|id_y|dept_y| age|
    +---+------+----+----+------+----+
    |  1|jordan|  DS|null|  null|null|
    |  2|  jack|  DS|null|  null|null|
    |  1|  jack| SDE|null|  null|null|
    |  2|  jack| SDE|null|  null|null|
    |  1|  jack|  PM|null|  null|null|
    |  2|  jack|  PM|null|  null|null|
    +---+------+----+----+------+----+

    '''
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix, 'left')
  
  def right_join(self, 
                 pyspark_df, 
                 on = None, 
                 on_x = None, 
                 on_y = None , 
                 sql_on = None, 
                 suffix = ["", "_y"]
                 ):
    '''
    Joins columns of pyspark_df to self by matching rows
    Includes all keys in pyspark_df

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.right_join(df2, on = ["id", "dept"]).show()
    +---+----+----+---+
    | id|dept|name|age|
    +---+----+----+---+
    |  2| SDE|jack| 10|
    |  1| SDE|jack| 20|
    |  2|  PM|jack| 30|
    |  2|  BA|null| 40|
    +---+----+----+---+

    >>> (df1.ts.right_join(df2, sql_on = '(LHS.id == RHS.id) & 
    >>>                    (LHS.dept == RHS.dept) & (RHS.age < 30)').show())
    +----+----+----+----+------+---+
    |  id|name|dept|id_y|dept_y|age|
    +----+----+----+----+------+---+
    |   2|jack| SDE|   2|   SDE| 10|
    |   1|jack| SDE|   1|   SDE| 20|
    |null|null|null|   2|    PM| 30|
    |null|null|null|   2|    BA| 40|
    +----+----+----+----+------+---+

    >>> df1.ts.right_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept']).show()
    # +----+----+----+----+------+---+
    # |  id|name|dept|id_y|dept_y|age|
    # +----+----+----+----+------+---+
    # |null|null|null|   2|   SDE| 10|
    # |null|null|null|   1|   SDE| 20|
    # |null|null|null|   2|    PM| 30|
    # |null|null|null|   2|    BA| 40|
    # +----+----+----+----+------+---+

    '''
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix, 'right')
  
  def inner_join(self, 
                 pyspark_df, 
                 on = None, 
                 on_x = None, 
                 on_y = None , 
                 sql_on = None, 
                 suffix = ["", "_y"]
                 ):
    '''
    Joins columns of pyspark_df to self by matching rows
    Includes only matching keys in pyspark_df and self

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.inner_join(df2, on = ["id", "dept"]).show()
    +---+----+----+---+
    | id|dept|name|age|
    +---+----+----+---+
    |  1| SDE|jack| 20|
    |  2|  PM|jack| 30|
    |  2| SDE|jack| 10|
    +---+----+----+---+

    >>> (df1.ts.inner_join(df2, sql_on = '(LHS.id == RHS.id) & 
    >>>                    (LHS.dept == RHS.dept) & (RHS.age < 30)').show())
    +---+----+----+----+------+---+
    | id|name|dept|id_y|dept_y|age|
    +---+----+----+----+------+---+
    |  1|jack| SDE|   1|   SDE| 20|
    |  2|jack| SDE|   2|   SDE| 10|
    +---+----+----+----+------+---+

    '''
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix, 'inner')
  
  def full_join(self, 
                pyspark_df, 
                on = None, 
                on_x = None, 
                on_y = None , 
                sql_on = None, 
                suffix = ["", "_y"]
                ):
    '''
    Joins columns of pyspark_df to self by matching rows
    Includes all keys from both pyspark_df and self

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.full_join(df2, on = ["id", "dept"]).show()
    +---+----+------+----+
    | id|dept|  name| age|
    +---+----+------+----+
    |  1|  DS|jordan|null|
    |  1|  PM|  jack|null|
    |  1| SDE|  jack|  20|
    |  2|  BA|  null|  40|
    |  2|  DS|  jack|null|
    |  2|  PM|  jack|  30|
    |  2| SDE|  jack|  10|
    +---+----+------+----+

    >>> (df1.ts.full_join(df2, sql_on = '(LHS.id == RHS.id) & 
    >>>                   (LHS.dept == RHS.dept) & (RHS.age < 30)').show())
    +----+------+----+----+------+----+
    |  id|  name|dept|id_y|dept_y| age|
    +----+------+----+----+------+----+
    |   1|jordan|  DS|null|  null|null|
    |   1|  jack|  PM|null|  null|null|
    |   1|  jack| SDE|   1|   SDE|  20|
    |null|  null|null|   2|    BA|  40|
    |   2|  jack|  DS|null|  null|null|
    |   2|  jack|  PM|null|  null|null|
    |null|  null|null|   2|    PM|  30|
    |   2|  jack| SDE|   2|   SDE|  10|
    +----+------+----+----+------+----+

    '''
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix, 'full')
  
  # alias
  outer_join = full_join
  
  def anti_join(self, 
                pyspark_df, 
                on = None, 
                on_x = None, 
                on_y = None , 
                sql_on = None
                ):
    '''
    Joins columns of pyspark_df to self by matching rows
    Includes keys in self if not present in pyspark_df

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    # Create the DataFrames
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.anti_join(df2, on = ["id", "dept"]).show()
    +---+----+------+
    | id|dept|  name|
    +---+----+------+
    |  1|  DS|jordan|
    |  2|  DS|  jack|
    |  1|  PM|  jack|
    +---+----+------+

    >>> (df1.ts.anti_join(df2, sql_on = '(LHS.id == RHS.id) & 
                          (LHS.dept == RHS.dept) & (RHS.age < 30)').show())
    +---+------+----+
    | id|  name|dept|
    +---+------+----+
    |  1|jordan|  DS|
    |  2|  jack|  DS|
    |  1|  jack|  PM|
    |  2|  jack|  PM|
    +---+------+----+

    >>> df1.ts.anti_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept']).show()
    +---+------+----+
    | id|  name|dept|
    +---+------+----+
    |  1|jordan|  DS|
    |  2|  jack|  DS|
    |  1|  jack| SDE|
    |  2|  jack| SDE|
    |  1|  jack|  PM|
    |  2|  jack|  PM|
    +---+------+----+
    '''
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on,
                     suffix = ["", "_y"], how = 'anti'
                     )
  
  def semi_join(self, 
                pyspark_df, 
                on = None, 
                on_x = None, 
                on_y = None , 
                sql_on = None
                ):
    '''
    Joins columns of pyspark_df to self by matching rows
    Includes keys in self if present in pyspark_df

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    on: string or a list of strings
      Common column names to match
    on_x: string or a list of strings
      Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
      Column names of y to be matched with arg 'on_x'
    sql_on: string
      SQL expression used to join both DataFrames. 
      Recommended for inequality joins.
      The left table has to be specified as 'LHS' and the right table as 'RHS'.
      e.g. '(LHS.dept == RHS.dept) & (LHS.age >= RHS.age) & (RHS.age < 30)'
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    # Create the DataFrames
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'DS'),
        (1, "jack", 'SDE'),
        (2, "jack", 'SDE'),
        (1, "jack", 'PM'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        (2, "PM", 30),
        (2, "BA", 40),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.semi_join(df2, on = ["id", "dept"]).show()
    # +---+----+----+
    # | id|dept|name|
    # +---+----+----+
    # |  1| SDE|jack|
    # |  2|  PM|jack|
    # |  2| SDE|jack|
    # +---+----+----+

    >>> (df1.ts.semi_join(df2, sql_on = '(LHS.id == RHS.id) & 
                          (LHS.dept == RHS.dept) & (RHS.age < 30)')).show()
    +---+------+----+
    | id|  name|dept|
    +---+------+----+
    |  1|jordan|  DS|
    |  2|  jack|  DS|
    |  1|  jack|  PM|
    |  2|  jack|  PM|
    +---+------+----+

    >>> df1.ts.semi_join(df2, on_x = ['id', 'dept'], on_y = ['age', 'dept']).show()
    # +---+----+----+
    # | id|name|dept|
    # +---+----+----+
    # +---+----+----+
    '''
     
    return self.join(pyspark_df, on, on_x, on_y, sql_on,
                     suffix = ["", "_y"], how = 'semi'
                     )
  
  def cross_join(self, pyspark_df, suffix = ["", "_y"]):
    '''
    Returns the cartesian product of two DataFrames.

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame)
    suffix: list of two strings
      suffix to append the columns of left and right in order to create
      unique names after the join

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    # Create the DataFrames
    >>> df1 = spark.createDataFrame([
        (1, "jordan", 'DS'),
        (2, "jack", 'PM')
        ],        
        ("id", "name", "dept")
    )

    >>> df2 = spark.createDataFrame([
        (2, "SDE", 20), 
        (1, "SDE", 10),
        ],
      ("id", "dept", "age")
    )

    >>> df1.ts.cross_join(df2).show()
    +---+------+----+---+----+---+
    | id| name|dept|id_y|dept_y|age_y|
    +---+------+----+---+----+---+
    |  1|jordan|  DS|  2| SDE| 20| 
    |  1|jordan|  DS|  1| SDE| 10|
    |  2|  jack|  PM|  2| SDE| 20|
    |  2|  jack|  PM|  1| SDE| 10|
    +---+------+----+---+----+---+
    '''
    
    return self._execute_cross_join(pyspark_df, suffix)

  # count methods ------------------------------------------------------------
  def count(self, column_names, name = 'n', wt = None):
    '''
    count
    Count unique combinations of columns

    Parameters
    ----------
    column_names : string or list of strings
      Names of columns
    name : string, optional
      Name of the count column to be created (should not be an existing
      name). The default is 'n'.
    wt: string, optional
      Name of the weight column. The default is None. When None, the number of
      rows is counted

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    pen.ts.count(['species', 'sex']).show()
    pen.ts.count('species', name = 'cnt').show()
    pen.ts.count('species', wt = 'body_mass_g').show()
    '''
    cn = self._validate_column_names(column_names)
    
    assert isinstance(name, str),\
      "'name' should be a string"
    assert name not in column_names,\
      "'name' should not be an element of 'column_names'"
    
    if wt is not None:
      assert isinstance(wt, str),\
        "'wt' should be a string"
      assert (wt in self.colnames) and (wt not in cn),\
        "'wt' should be an existing column name and not in 'column_names'"
    
    if wt is None:
      res = (self.__data
                 .select(*cn)
                 .withColumn(name, F.lit(1))
                 .groupby(*cn)
                 .agg(F.sum(F.col(name)).alias(name))
                 )
    else:
      cn_with_wt = cn + [wt]
      res = (self.__data
                 .select(*cn_with_wt)
                 .groupby(*cn)
                 .agg(F.sum(F.col(wt)).alias(name))
                 )
      
    return res

  def add_count(self, column_names, name = 'n', wt = None):
    '''
    add_count
    Add a column of counts of unique combinations of columns
  
    Parameters
    ----------
    column_names : string or list of strings
      Names of columns
    name : string, optional
      Name of the count column to be created (should not be an existing
      name). The default is 'n'.
    wt: string, optional
      Name of the weight column. The default is None. When None, the number of
      rows is counted
      
    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.add_count(['species', 'sex']).show()
    >>> pen.ts.add_count('species', name = 'cnt').show()
    >>> pen.ts.add_count('species', wt = 'body_mass_g').show()
    '''
    cn = self._validate_column_names(column_names)
    
    assert isinstance(name, str),\
      "'name' should be a string"
    assert name not in self.colnames,\
      "'name' should not be an existing column name"
    
    if wt is not None:
      assert isinstance(wt, str),\
        "'wt' should be a string"
      assert (wt in self.colnames) and (wt not in cn),\
        "'wt' should be an existing column name and not in 'column_names'"
    
    if wt is None:
      res = (self.__data
                 .select(*cn)
                 .withColumn(name, F.lit(1))
                 .groupby(*cn)
                 .agg(F.sum(F.col(name)).alias(name))
                 )
    else:
      cn_with_wt = cn + [wt]
      res = (self.__data
                 .select(*cn_with_wt)
                 .groupby(*cn)
                 .agg(F.sum(F.col(wt)).alias(name))
                 )
    
    res = (self.__data
               .join(res, on = cn, how = "left")
               )
    return res
  
  # pipe methods -------------------------------------------------------------
  def pipe(self, func, *args, **kwargs):
    '''
    pipe
    returns func(self, ...)
    
    Parameters
    ----------
    func: callable
      function to call
    *args : args
    **kwargs : kwargs
    
    Return
    ------
    Depends on func
    '''
    assert callable(func)
    return func(self.__data, *args, **kwargs)
  
  def pipe_tee(self, func, *args, **kwargs):
    '''
    pipe_tee
    use pipe for side-effect and return input

    Parameters
    ----------
    func: callable
      function to call
    *args : args
    **kwargs : kwargs

    Returns
    -------
    Input pyspark dataframe
    '''
    assert callable(func)
    func(self.__data, *args, **kwargs) # side-effect
    return self.__data
  
  # slice min and max methods -----------------------------------------------
  def slice_min(self,
                n,
                order_by_column,
                with_ties = True, 
                by = None
                ):
    '''
    slice_min
    Subset top rows ordered by some column

    Parameters
    ----------
    n : int
      Number of rows to subset
    order_by_column : string
      Name of the column to order by in ascending nulls to last
    with_ties : bool, optional
      Whether to return all rows when ordering results in ties.
      The default is True.
    by : string or list of strings, optional
      column(s) to group by. The default is None.

    Returns
    -------
    pyspark dataframe
    
    Details
    -------
    The ordering always keeps null to last
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.slice_min(n = 2,
    >>>                  order_by_column = 'bill_depth_mm',
    >>>                  with_ties = False,
    >>>                  by = ['species', 'sex']
    >>>                  )
    '''
    
    assert isinstance(n, int) and n > 0,\
      "n should be a positive integer"
    order_by_spec = self._validate_order_by((order_by_column, 'asc'))
    assert isinstance(with_ties, bool),\
      "'with_ties' should be a bool"
    
    # create windowspec
    if by is None:
      win = self._create_windowspec(order_by = order_by_spec)
    else:
      by = self._validate_by(by)
      win = self._create_windowspec(order_by = order_by_spec, by = by)
    
    # decide on ranking function
    if with_ties:
      rank_func = F.dense_rank()
    else:
      rank_func = F.row_number()
    
    # core
    res = (self.__data
               .withColumn('_rn', rank_func.over(win))
               .filter(F.col('_rn') <= n)
               .drop('_rn')
               )
    
    return res
    
  def slice_max(self,
                n,
                order_by_column,
                with_ties = True, 
                by = None
                ):
    '''
    slice_max
    Subset top rows ordered by some column

    Parameters
    ----------
    n : int
      Number of rows to subset
    order_by_column : string
      Name of the column to order by in descending nulls to first
    with_ties : bool, optional
      Whether to return all rows when ordering results in ties.
      The default is True.
    by : string or list of strings, optional
      column(s) to group by. The default is None.

    Returns
    -------
    pyspark dataframe
    
    Details
    -------
    The ordering always keeps null to last
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.slice_max(n = 2,
    >>>                  order_by_column = 'bill_depth_mm',
    >>>                  with_ties = False,
    >>>                  by = ['species', 'sex']
    >>>                  )
    '''
    assert isinstance(n, int) and n > 0,\
      "n should be a positive integer"
    order_by_spec = self._validate_order_by((order_by_column, 'desc'))
    assert isinstance(with_ties, bool),\
      "'with_ties' should be a bool"
    
    # create windowspec
    if by is None:
      win = self._create_windowspec(order_by = order_by_spec)
    else:
      by = self._validate_by(by)
      win = self._create_windowspec(order_by = order_by_spec, by = by)
    
    # decide on ranking function
    if with_ties:
      rank_func = F.dense_rank()
    else:
      rank_func = F.row_number()
    
    # core
    res = (self.__data
               .withColumn('_rn', rank_func.over(win))
               .filter(F.col('_rn') <= n)
               .drop('_rn')
               )
    
    return res

  # rbind --------------------------------------------------------------------
  def rbind(self, pyspark_df, id = None):
    '''
    rbind
    bind or concatenate rows of two dataframes

    Parameters
    ----------
    pyspark_df : pyspark dataframe
    id : str, optional
      When not None, a id column created with value 'left' for self and
      'right' for the pyspark_df.
      The default is None.

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> df1 = pen.select(['species', 'island', 'bill_length_mm'])
    >>> df2 = pen.select(['species', 'island', 'bill_depth_mm'])
    >>> 
    >>> df1.ts.rbind(df2).show()
    >>> df1.ts.rbind(df2, id = "id").show()
    '''
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
      "'pyspark_df' should be a pyspark dataframe"
    
    LHS = self.__data
    RHS = pyspark_df
    
    if id is not None:
      assert isinstance(id, str)
      cn = list(set(LHS.columns).union(RHS.columns))
      assert id not in cn,\
        "id should not be a column name of one of the two dataframes"
      res = (LHS.withColumn(id, F.lit('left'))
                .unionByName(RHS.withColumn(id, F.lit('right')),
                             allowMissingColumns = True
                             )
                )
    else:
      res = LHS.unionByName(RHS, allowMissingColumns = True)
    
    if id is not None:
      non_id_cols = list(set(res.columns).difference([id]))
      res = res.select([id] + non_id_cols)
    
    return res
  
  # alias
  bind_rows = rbind
  
  # union --------------------------------------------------------------------
  def union(self, pyspark_df):
    '''
    rbind
    bind or concatenate rows of two dataframes

    Parameters
    ----------
    pyspark_df : pyspark dataframe
    id : str, optional
      When not None, a id column created with value 'left' for self and
      'right' for the pyspark_df.
      The default is None.

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> df1 = pen.ts.slice_max(n = 2,
    >>>                        order_by_column = 'bill_length_mm',
    >>>                        with_ties = False
    >>>                        )
    >>> df2 = pen.ts.slice_max(n = 4,
    >>>                        order_by_column = 'bill_length_mm',
    >>>                        with_ties = False
    >>>                        )
    >>> 
    >>> df1.ts.union(df2).show()
    '''
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
      "'pyspark_df' should be a pyspark dataframe"
    
    res = (self.__data
               .unionByName(pyspark_df, allowMissingColumns = True)
               .distinct()
               )
    return res
  
  # pivot methods ------------------------------------------------------------
  def pivot_wider(self, 
                  names_from,
                  values_from,
                  values_fill = None,
                  values_fn = None, 
                  id_cols = None,
                  sep = "__",
                  names_prefix = "",
                  names_expand = False,
                  ):
    '''
    Pivot data from long to wide
    
    Parameters
    ----------
    names_from: string or list of strings
      column names whose unique combinations are expected to become column
      new names in the result
    values_from: string or list of strings
      column names to fill the new columns with
    values_fill: scalar, list or dict (default is None)
      Optionally, a (scalar) value that specifies what each value should be
      filled in with when missing.
      This can be a dictionary if you want to apply different fill values to
      different value columns.
      Make sure only compatible data types are used in conjunction with 
      the pyspark column.
      Missing values can only be filled with a scalar, or empty python
      list value.
    
    values_fn: string(of pyspark functions) or 
               a dict of strings(of pyspark funtions) (default is None)
      A function to handle multiple values per row in the result.
      When a dict, keys should be a subset of arg 'values_from'.
      F.collect_list is applied by default in case nothing is specified.
      The string pf pyspark functions should be passed as a string 
      starting with 'F.'
      This is to indicate that the function is from pyspark.
      Unlist the pivot columns to scalar values if and only if:
      1. values_fn is None and values_fn is None
      2. all the pivot columns are of type list/array and all the pivot_cols
         have values with a maximum length of 1.
    
    id_cols: string or list of strings, default is None
      Names of the columns that should uniquely identify an observation 
      (row) after widening (columns of the original dataframe that are
      supposed to stay put)
        
    sep: string (default is "__")
      seperator to use while creating resulting column names
        
    names_prefix: string (default is "")
      prefix for new columns

    names_expand: boolean (default is False)
      When True, he output will contain column names corresponding 
      to a complete expansion of all possible values in names_from. 
      Implicit factor levels that aren't represented in the data will
      become explicit. 
        
    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> pen.ts.pivot_wider(id_cols = "island",
    >>>                    names_from  = "sex,
    >>>                    values_from = "bill_length_mm"
    >>>                   )
    >>> # All three inputs: 'id_cols', 'names_from', 'values_from' can be lists
    >>> pen.ts.pivot_wider(id_cols = ["island", "sex"], 
    >>>                   names_from  = "species",
    >>>                   values_from = "bill_length_mm"
    >>>                   )
    >>> pen.ts.pivot_wider(
    >>>     id_cols       = ["island", "sex"]
    >>>     , names_from  = "species"
    >>>     , values_from = ["bill_length_mm", "bill_depth_mm"]
    >>>     )
    >>> pen.ts.pivot_wider(id_cols = ["island", "sex"]
    >>>                           , names_from  = ["species", "year"]
    >>>                           , values_from = "bill_length_mm"
    >>>                           )
    >>> pen.ts.pivot_wider(
    >>>     id_cols       = ["island", "sex"]
    >>>     , names_from  = ["species", "year"]
    >>>     , values_from = ["bill_length_mm", "bill_depth_mm"]
    >>>     )
    >>> # when id_cols is empty, all columns except the columns from
    >>> # `names_from` and `values_from` are considered as id_cols
    >>> (pen.ts.select(['flipper_length_mm', 'body_mass_g'], include = False)
    >>>         .pivot_wider(names_from    = ["species", "year"]
    >>>               , values_from = ["bill_length_mm", "bill_depth_mm"]
    >>>               )
    >>>  )
    >>> # use some prefix for new columns
    >>> pen.ts.pivot_wider(id_cols       = "island"
    >>>                           , names_from  = "sex"
    >>>                           , values_from = "bill_length_mm"
    >>>                           , names_prefix = "gender_"
    >>>                           )
    '''
    
    cn = self.colnames
    
    names_from = self._validate_column_names(names_from)
    values_from = self._validate_column_names(values_from)

    assert len(set(values_from).intersection(names_from)) == 0,\
      ("arg 'names_from' and 'values_from' should not "
       "have common column names"
       )
    names_values_from = set(values_from).union(names_from)
    
    if id_cols is None:
      id_cols = list(set(cn).difference(names_values_from))
      if len(id_cols) == 0:
        raise Exception(
          ("'id_cols' is turning out to be empty. Choose the "
           "'names_from' and 'values_from' appropriately or specify "
           "'id_cols' explicitly."
           ))
      else:
          print("'id_cols' chosen: " + str(id_cols))
    else:
      id_cols = self._validate_column_names(id_cols)
      assert len(set(id_cols).intersection(names_values_from)) == 0,\
        ("arg 'id_cols' should not have common names with either "
         "'names_from' or 'values_from'"
         )
        
    assert (values_fn is None
            or isinstance(values_fn, dict)
            or isinstance(values_fn, str)
            ),\
      "arg 'values_fn' should be a string or a dictionary of strings."
    
    if isinstance(values_fn, str):
      assert values_fn.startswith('F.'),\
        ("arg 'values_fn' should be a string starting with 'F.'. "
         " It indicates a pyspark function.")
    elif isinstance(values_fn, dict):
      assert set(values_fn.keys()).issubset(set(values_from)),\
        ("arg 'values_fn' should be a dictionary with keys as a subset "
         "of 'values_from'")
      assert all([fn.startswith('F.') for fn in values_fn.values()]),\
        ("arg 'values_fn' should be a dictionary of strings starting with "
         "'F.' It indicates a pyspark function.")
      
    if isinstance(values_fill, dict):
      assert set(values_fill.keys()).issubset(set(values_from)),\
        ("arg 'values_fill' should be a dictionary with keys as a subset "
         "of 'values_from'")
    
    assert isinstance(sep, str),\
      "arg 'sep' should be a string"
        
    assert isinstance(names_prefix, str),\
      "arg 'names_prefix' should be a string without spaces"
    assert ' ' not in names_prefix,\
      "arg 'names_prefix' should be a string without spaces"
    
    assert isinstance(names_expand, bool),\
      "arg 'names_expand' should be a boolean"
    
    df = self.__data

    # Construct the pyspark expression for the pivot_col via names_from
    # e.g. [Column<'name'>, Column<'name2'>] for names_from = ['name', 'name2']
    names_from_pyspark_expr = [F.col(name) for name in names_from]

    # Create the pivot column in the dataframe via names_from_pyspark_expr
    # e.g Column<'name__name2'> for names_from = ['name', 'name2'],
    # where sep = '__'
    df = df.withColumn('pivot_col', F.concat_ws(sep, *names_from_pyspark_expr))

    # Construct the pivot columns via names_from and names_from_pyspark_expr
    # When names_expand = True, it will return all possible combinations of
    # values in names_from
    # When names_expand = False, it will return the unique values in names_from
    pivot_cols = self._get_pivot_columns(names_from, sep, names_expand,
                                         df, names_from_pyspark_expr
                                         )

    # Construct the pyspark expression for the values_from.
    # e.g. [Column<'avg(dept) AS dept'>,
    # Column<'collect_list(dept2) AS dept2'>] 
    # for values_from = ['dept', 'dept2'] and 
    #     values_fn = {'dept': 'avg', 'dept2': 'collect_list'}
    values_from_pyspark_expr = ( 
      self._construct_pyspark_expr_from_values_from(values_from, values_fn))

    # Core logic for pivot_wider
    df = (df.groupBy(id_cols)
            .pivot('pivot_col', pivot_cols)
            .agg(*values_from_pyspark_expr)
            )

    # Get the new pivot columns
    new_pivot_cols = list(set(df.columns).difference(id_cols))

    # Fill missing values in the pivot columns.
    df = self._fill_missing_values_for_pivot_columns(values_fill,
                                                     df,
                                                     new_pivot_cols
                                                     )

    # Unlist the pivot columns to scalar values if and only if:
    # 1. values_fn is None and values_fn is None
    # 2. all the pivot columns are of type list/array and all the pivot_cols
    #    have values with a maximum length of 1.
    df = self._unlist_pivot_cols(values_fill, values_fn, df, new_pivot_cols)    

    # Replace empty lists or sets with None
    for col in new_pivot_cols:
      col_dtype = df.select(col).dtypes[0][1]
      if col_dtype.startswith('array') or col_dtype.startswith('map'):
        df = df.withColumn(col,
                           F.when(F.size(col) > 0,
                                  F.col(col)
                                  ).otherwise(None)
                           )

    # Rename the pivot columns in case names_prefix is passed explicitly.
    if names_prefix is not None and  names_prefix != "":
      select_expr = ([F.col(col).alias(names_prefix + sep + col) 
                      if col in new_pivot_cols
                      else col 
                      for col in df.columns
                      ])
      df = df.select(select_expr)
    return df

  def pivot_longer(self, 
                  cols,
                  names_to = "name",
                  values_to = "value",
                  include = True,
                  values_drop_na = False
                  ):
    '''
    Pivot from wide to long
    aka melt
    
    Parameters
    ----------
    cols: list of strings
      Column names to be melted.
      The dtypes of the columns should match.
      Leftover columns are considered as 'id' columns.
      When include is False, 'cols' refers to leftover columns.
    names_to: string (default: 'name')
      Name of the resulting column which will hold the names of the columns
      to be melted
    values_to: string (default: 'value')
      Name of the resulting column which will hold the values of the columns
      to be melted
    include: bool (default: True)
      If True, cols are used to melt. Else, cols are considered as 'id'
      columns and the leftover columns are melted.
    values_drop_na: bool (default: False)
      Whether to drop the rows corresponding to missing value in the result

    Returns
    -------
    pyspark dataframe
    
    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> df = (pen.ts.select(['species', 
    >>>                     'bill_length_mm', 
    >>>                     'bill_depth_mm', 
    >>>                     'flipper_length_mm']
    >>>                     )
    >>>       )
    >>> df.pivot_longer(cols = ['bill_length_mm', 
    >>>                         'bill_depth_mm',
    >>>                         'flipper_length_mm']
    >>>                 )
    >>> # pivot by specifying 'id' columns to obtain the same result as above
    >>> # this is helpful when there are many columns to melt
    >>> df.pivot_longer(cols = 'species',
    >>>                include = False
    >>>                )
    >>> # If you want to drop the rows corresponding to missing value in the result,
    >>> # set values_drop_na to True
    >>> df.pivot_longer(cols = ['bill_length_mm',
    >>>                         'bill_depth_mm'],
    >>>                 values_drop_na = True
    >>>                 )            
    '''
    # assertions
    cn = self.colnames
    cols = self._validate_column_names(cols)

    assert isinstance(include, bool),\
      "arg 'include' should be a bool"
    if not include:
      cols = list(setlist(cn).difference(cols))
      assert len(cols) > 0,\
        "At least one column should be selected for melt"
    
    id_vars = set(cn).difference(cols)
    assert isinstance(names_to, str),\
      "arg 'names_to' should be a string"
    assert isinstance(values_to, str),\
      "arg 'values_to' should be a string"
    assert names_to not in id_vars,\
      "arg 'names_to' should not match a id column"
    assert values_to not in id_vars,\
      "arg 'values_to' should not match a id column"
    assert isinstance(values_drop_na, bool),\
      "arg 'values_drop_na' should be a bool"

    # Bulding the pyspark expression for melt.
    # Sample pyspark expression: stack(3,
    #                                 'bill_length_mm', `bill_length_mm`,
    #                                 'bill_depth_mm', `bill_depth_mm`,
    #                                 'flipper_length_mm', `flipper_length_mm`
    #                                 ) as (`name`, `value`)
    
    melt_cols = ', '.join([f"'{col}', `{col}`" for col in cols])
    melt_expr = (f"stack({len(cols)}, {melt_cols}) as (`{names_to}`, "
                 f"`{values_to}`)")
  
    # Melt/ pivot_longer core logic.
    res = self.__data.select(*id_vars, F.expr(melt_expr))

    # Drop rows with null values in the values_to column
    if values_drop_na:
      res = res.filter(F.col(values_to).isNotNull())

    return res
  
  def _construct_pyspark_expr_from_values_from(self, values_from, values_fn):
      
    if values_fn is None:
      
      # Since values_fn is None, we do a just a simple collect_list.
      # e.g. [Column<'collect_list(dept) AS dept'>,
      #       Column<'collect_list(dept2) AS dept2'>] 
      # for values_from = ['dept', 'dept2'] and values_fn = None
      values_from_pyspark_expr = ([F.collect_list(F.col(vf)).alias(vf) 
                                   for vf in values_from]
                                  )
    elif isinstance(values_fn, str):
      
      # Apply the function name passed in values_fn to each column in
      # values_from.
      # e.g [Column<'sum(salary) AS salary'>,
      #      Column<'sum(salary2) AS salary2'>] 
      # for values_from = ['salary', 'salary2'] and values_fn = 'sum'
      values_from_pyspark_expr = (
        [eval(f"{values_fn}(F.col('{vf}')).alias('{vf}')")
         for vf in values_from
         ])
    elif isinstance(values_fn, dict):
      values_from_pyspark_expr = []
      for vf in values_from:
        # If the column name is not present in the dictionary, 
        # we do a simple collect_list.
        # Else, we use the function name passed in the dictionary.
        # e.g. [Column<'collect_list(dept) AS dept'>,
        #       Column<'sum(salary) AS salary'>] 
        # for values_from = ['dept', 'salary'] and 
        #     values_fn = {'salary': 'sum'}
        func_expr = values_fn.get(vf, 'F.collect_list')
        values_from_pyspark_expr.append(
          eval(f"{func_expr}(F.col('{vf}')).alias('{vf}')"))

    return values_from_pyspark_expr
  
  def _get_pivot_columns(self, names_from, sep, names_expand, df,
                         names_from_pyspark_expr):
      
    # when names_expand is True, all unique combinations of names_from 
    # columns are taken
    # even if they are not present in the original dataframe
    if names_expand:
      # cartesian product of names_from
      cartesian_product_names_from_df = df.select(names_from[0]).distinct()
      if len(names_from) > 1:
        for i in range(1, len(names_from)):
          distinct_values_df = df.select(names_from[i]).distinct()
          cartesian_product_names_from_df = (
            cartesian_product_names_from_df.crossJoin(distinct_values_df)
            )
    
      # Construct the pivot_col from names_from
      cartesian_product_names_from_df = (
        cartesian_product_names_from_df.withColumn(
          'pivot_col',
          F.concat_ws(sep, *names_from_pyspark_expr)
          )
        )

      # Construct the unique combinations of names_from columns in a 
      # python list
      pivot_cols = ([row.pivot_col 
                     for row in cartesian_product_names_from_df
                                .select('pivot_col').
                                collect()]
      )
    else:
      pivot_cols = ([row.pivot_col 
                     for row in df.select('pivot_col').distinct().collect()]
      )    
      
    return pivot_cols
  
  def _fill_missing_values_for_pivot_columns(self, values_fill,
                                             df, new_pivot_cols
                                             ):
      
    # values_fill = {'dept': 'fill', 'dept2': []}
    # Construct a dictionary to fill the null values in the pivot columns.
    # The dictionary will only contain scalar values.
    new_values_fill = {}

    if values_fill is not None:
      if isinstance(values_fill, dict):
        for col in values_fill:
          # e.g cols_ending_with_col = ['jack_dept', 'jordan_dept']
          # for col = 'dept'
          new_cols_ending_with_pivot_col = ([c for c in new_pivot_cols
                                             if c.endswith('_' + col)])
          # e.g new_values_fill['jack_dept'] = 'fill' for col = 'dept'
          for new_col in new_cols_ending_with_pivot_col:
            missing_value_to_be_filled = values_fill[col]
            # Fill the null values in the pivot columns with the empty list.
            if isinstance(missing_value_to_be_filled, list):
              df = df.withColumn(new_col,
                                F.when(F.col(new_col).isNull(), 
                                       F.array()
                                       ).otherwise(F.col(new_col))
                                )
            elif np.isscalar(missing_value_to_be_filled):
              # Add the scaler value to the dictionary.
              new_values_fill[new_col] = values_fill[col]              
      elif isinstance(values_fill, list):
        
        # Fill the null values in the pivot columns with the empty list.
        for new_pivot_col in new_pivot_cols:
          df = df.withColumn(new_pivot_col, 
                             F.when(F.col(new_pivot_col).isNull(), 
                                    F.array()
                                    ).otherwise(F.col(new_pivot_col))
                             )
      elif np.isscalar(values_fill):
        new_values_fill = {col: values_fill for col in new_pivot_cols}
     
      # Fill the null values in the pivot columns 
      # with the new renamed dictinary.
      df = df.na.fill(new_values_fill)
    
    return df
  
  def _unlist_pivot_cols(self, values_fill, values_fn, df, new_pivot_cols):
      
    if values_fill is None and values_fn is None:
      # Get the data types of the pivot columns.
      col_dtypes = [df.select(col).dtypes[0][1] for col in new_pivot_cols]

      # If all the pivot columns are of type array, 
      # then proceed to convert them to scalars 
      # if subsequent conditions are met.
      if all([dtype.startswith('array') for dtype in col_dtypes]):
        # Check if all the pivot columns of type array 
        # have a maximum of only one element.
        is_column_scalar = [(df.select(pivot_col)
                              .filter(~F.isnull(pivot_col))
                              .filter(F.size(pivot_col) > 1)
                              .count()
                            ) == 0 for pivot_col in new_pivot_cols]
        # If all the pivot columns of type array have a maximum of 
        # only one element, unlist them.
        if all(is_column_scalar):
          for pivot_col in new_pivot_cols:
            df = df.withColumn(pivot_col, F.col(pivot_col).getItem(0))
            
    return df
  
  # nest and unnest ----------------------------------------------------------
  def nest_by(self, by, name = "data"):
    '''
    nest
    nest a pyspark dataframe per group as an array of structs

    Parameters
    ----------
    by : string or list of strings, optional
      column(s) to group by.
      
    name : string
      Name of the nested column which will be created.

    Returns
    -------
    res : pyspark dataframe
    '''
    by = self._validate_by(by)
    other_cols_list = list(set(self.colnames).difference(by))
    assert isinstance(name, str),\
      "name should be a string"
    assert name not in by,\
      "name should not be one of the 'by' column names"
    
    res = (self.__data
               .groupby(by)
               .agg(F.collect_list(F.struct(*other_cols_list)).alias(name))
               )
    
    return res
  
  # alias for nest_by
  nest = nest_by
  
  def unnest_wider(self, colname):
    '''
    unnest_wider
    creates multiple columns from a struct column

    Parameters
    ----------
    colname : string
      Name of the column of struct type.

    Returns
    -------
    pyspark dataframe
    '''
    colname = self._validate_column_names(colname)[0]
    schema  = self.__data.select(colname).schema
    
    # is colname column a struct?
    coltype = [x.dataType.typeName() for x in schema.fields][0]
    assert coltype == 'struct',\
      "schema of 'colname' column should be of type struct"
    
    # get names and types of structFields (columns within struct)
    json_obj        = json.loads(schema[colname].json())
    json_obj_simple = json_obj['type']['fields']
    names_column    = [x['name'] for x in json_obj_simple]
    
    # are colnames within struct unique?
    assert _is_unique_list(names_column),\
      "names in colname struct should not be duplicated"
    rest_columns = list(set(self.colnames).difference(colname))
    
    assert len(set(names_column).intersection(rest_columns)) == 0,\
      ("unnest_wider results in duplicate column names. "
       "Try renaming columns other than colname")
    
    res = (self.__data
              .select('*', colname + '.*')
              .drop(colname)
              )
    
    return res
  
  def unnest_longer(self, colname, name = "name", value = "value"):
    '''
    unnest_longer
    creates key and value columns from a struct column

    Parameters
    ----------
    colname : str
      Name of the column of struct type.
    name: str
      Name of the resulting key column. Default is 'name'.
    value: str
      Name of the resulting value column. Default is 'value'.

    Returns
    -------
    pyspark dataframe
    '''
    colname = self._validate_column_names(colname)[0]
    schema  = self.__data.select(colname).schema
    
    # is colname column a struct?
    coltype = [x.dataType.typeName() for x in schema.fields][0]
    assert coltype == 'struct',\
      "schema of colname column should be of type struct"
    
    # get names and types of structFields (columns within struct)
    json_obj        = json.loads(schema[colname].json())
    json_obj_simple = json_obj['type']['fields']
    names_column    = [x['name'] for x in json_obj_simple]
    
    # are colnames within struct unique?
    assert _is_unique_list(names_column),\
      "names in colname struct should not be duplicated"
    
    # name and value column should not intersect with id columns
    other_colnames = list(set(self.colnames).difference([colname]))
    assert isinstance(name, str),\
      "'name' should be a string"
    assert isinstance(value, str),\
      "'value' should be a string"
    assert not any([x in other_colnames for x in [name, value]]),\
      ("'name' and 'value' should be different from columns "
       " other than 'colname'")
    
    # rename non struct columns by prepending 'retain_'
    sel = [x + " as " + ("retain_" + x) for x in other_colnames]
    rev_sel = [("retain_" + x) + " as " + x for x in other_colnames]
    retained_colnames = [("retain_" + x) for x in other_colnames]
    
    # unpivot str
    # proto: "stack(2, 'Canada', Canada, 'China', China) as (Country,Total)"
    col_str = ["'" + x + "'," + x + "," for x in names_column]
    col_str = "".join(col_str)
    col_str = col_str[0:-1]
    
    unpivot_str = ("stack(" 
                   + str(len(names_column)) 
                   + ","
                   + col_str
                   + ") as ("
                   + name + ", " + value + ")"
                   )
    
    res = (self.__data
               # rename id cols to start with "retain_"
               # selectExpr is required instead of select
               .selectExpr(sel + [colname])
               # expand struct column
               .select('*', colname + '.*')
               .drop(colname)
               # unpivot or pivot longer
               .select(*retained_colnames, F.expr(unpivot_str))
               # drop rows where value is null
               .filter(f'{value} is not null')
               # rename "retain" columns
               .selectExpr(rev_sel + [name, value])
               )
    
    return res
    
  
  def unnest(self, colname):
    '''
    unnest
    unnests an column where each element is an array of structs

    Parameters
    ----------
    colname : string
      Name of the column of struct type.

    Returns
    -------
    pyspark dataframe
    '''
    colname = self._validate_column_names(colname)[0]
    schema  = self.__data.select(colname).schema
    
    # is colname column a array?
    coltype = [x.dataType.typeName() for x in schema.fields][0]
    assert coltype == 'array',\
      "schema of colname column should be of type array"
    
    # explode the array column
    res = (self.__data
               .select('*', F.explode_outer(colname).alias("_exploded"))
               .drop(colname)
               .withColumnRenamed("_exploded", colname)
               )
    
    # if array resolves into a regular column after explode, then we are done.
    # Else, if it is a struct, widen it
    res_names = res.columns
    res_types = [x.dataType.typeName() for x in res.schema.fields]
    res_dict = dict(zip(res_names, res_types))
    
    if res_dict[colname] == "struct":
      res = (res.select('*', colname + '.*')
                .drop(colname)
                )
    
    return res
  
  def fill_na(self, column_direction_dict, order_by, by = None):
    '''
    fill_na (alias: fill)
    fill missing values from neighboring rows

    Parameters
    ----------
    column_direction_dict : dict
      key is column. value should be one among:
      "up", "down", "updown", "downup" 
    order_by : string, tuple or list of tuples
      order_by specification
    by : string or list of strings, optional
      Names of columns to partition by. The default is None.

    Returns
    -------
    pyspark dataframe

    '''
    assert isinstance(column_direction_dict, dict)
    valid_values = ["up", "down", "updown", "downup"]
    assert set(column_direction_dict.values()).issubset(valid_values),\
      f"Values of column_direction_dict should be one among {valid_values}"
    _ = self._validate_column_names(list(column_direction_dict.keys()))
    
    # create two windowspec depending on direction
    order_by = self._validate_order_by(order_by)
    if by is None:
      win_down = self._create_windowspec(order_by = order_by,
                                         rows_between = [-sys.maxsize, 0]
                                         )
      win_up   = self._create_windowspec(order_by = order_by,
                                         rows_between = [0, sys.maxsize]
                                         )
    else:
      by = self._validate_by(by)
      win_down = self._create_windowspec(order_by = order_by,
                                         by = by,
                                         rows_between = [-sys.maxsize, 0]
                                         )
      win_up   = self._create_windowspec(order_by = order_by,
                                         by = by,
                                         rows_between = [0, sys.maxsize]
                                         )
    
    # first round of filling -- up or down
    col_expr_dict = dict()
    for col_name, direction in column_direction_dict.items():
      if direction in ["down", "downup"]:
        col_expr_dict[col_name] = F.last(F.col(col_name),
                                          ignorenulls = True
                                          ).over(win_down)
      elif direction in ["up", "updown"]:
        col_expr_dict[col_name] = F.first(F.col(col_name),
                                         ignorenulls = True
                                         ).over(win_up)
        
    # 2nd round of filling -- down'up' or up'down' (focus things in quote)
    col_expr_dict2 = dict()
    for col_name, direction in column_direction_dict.items():
      if direction == "updown":
        col_expr_dict2[col_name] = F.last(F.col(col_name),
                                           ignorenulls = True
                                           ).over(win_down)
      elif direction == "downup":
        col_expr_dict2[col_name] = F.first(F.col(col_name),
                                          ignorenulls = True
                                          ).over(win_up)
    
    res = self.__data.withColumns(col_expr_dict)
    if len(col_expr_dict2) > 0:
      res = res.withColumns(col_expr_dict2)
    return res
  
  # alias for fill_na
  fill = fill_na

  def drop_na(self, 
              column_names = None, 
              how = "any",
              thresh = None
              ):

    '''
    drop_na (alias: drop)
    drop rows with null values

    Parameters
    ----------
    how : string, optional
      "any" or "all". The default is "any".
      If 'any', drop a row if it contains any nulls. 
      If 'all', drop a row only if all its values are null.

    column_names : string or list of strings, optional
      It specifies the columns to consider for null values. 
      If a row has null values only in the specified columns,
      it will be dropped.

    thresh : int, optional
      Number of non-null values required to keep a row. The default is None.
      This overrides how parameter.

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> # create a DataFrame with null values
    >>> data = [("Alice", 25, None), ("Bob", None, 80), (None, 30, 90)]
    >>> df = spark.createDataFrame(data, ["name", "age", "score"])
    >>> # drop rows with null values
    >>> df1 = df.ts.drop_na()
    +----+---+-----+
    |name|age|score|
    +----+---+-----+
    +----+---+-----+
    >>> # drop rows with null values in a specific column
    >>> df2 = df.ts.drop_na(column_names = ["age"])
    +-----+---+-----+
    | name|age|score|
    +-----+---+-----+
    |Alice| 25| null|
    | null| 30|   90|
    +-----+---+-----+
    >>> # drop rows with null values if all values are null.
    >>> df3 = df.ts.drop_na(how = "all")
    +-----+----+-----+
    | name| age|score|
    +-----+----+-----+
    |Alice|  25| null|
    |  Bob|null|   80|
    | null|  30|   90|
    +-----+----+-----+
    >>> # drop rows with less than 3 non-null values
    >>> df4 = df.ts.drop_na(thresh=3)
    +----+---+-----+
    |name|age|score|
    +----+---+-----+
    +----+---+-----+
    '''
    assert how in ["any", "all"],\
      "how should be one among 'any' or 'all'"
    
    assert thresh is None or isinstance(thresh, int),\
      "thresh should be an integer if specified"
    
    if column_names is not None:
      column_names = self._validate_column_names(column_names)

    res = self.__data.dropna(subset = column_names,
                             how = how,
                             thresh = thresh,
                             )
    return res

  # alias for drop_na
  drop = drop_na

  def replace_na(self,value):
    '''
    Replace missing values with a specified value.

    Parameters
    ----------
    value: dict or a scalar or an empty list
      When a dict, key should be a column name and value should be the
      value to replace by missing values of the column
      When a scalar or an empty list, 
      missing values of all columns will be replaced with value. 
      A scalar value could be a string, a numeric value(int, float), 
      or a boolean value.

    Returns
    -------
    pyspark dataframe

    Examples
    --------
    >>> import tidypyspark.tidypyspark_class as ts
    >>> from tidypyspark.datasets import get_penguins_path
    >>> pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)
    >>> # create a DataFrame with null values
    >>> data = [("Alice", 25, None, [20, 30, 40]), 
    >>>         ("Bob", None, 80, [10, 20, 30]), 
    >>>         (None, 30, 90 , None)
    >>>         ]
    >>> df = spark.createDataFrame(data, ["name", "age", "score", "marks"])
    +-----+----+-----+------------+
    | name| age|score|       marks|
    +-----+----+-----+------------+
    |Alice|  25| null|[20, 30, 40]|
    |  Bob|null|   80|[10, 20, 30]|
    | null|  30|   90|        null|
    +-----+----+-----+------------+
    >>> # replace null values with a dictionary of column names and values
    >>> df2 = df.ts.replace_na({"name": "A", "score": 25, "marks": []})
    +-----+----+-----+------------+
    | name| age|score|       marks|
    +-----+----+-----+------------+
    |Alice|  25|   25|[20, 30, 40]|
    |  Bob|null|   80|[10, 20, 30]|
    |    A|  30|   90|          []|
    +-----+----+-----+------------+
    '''
    df = self.__data

    # Get the datatypes of the columns in the dataframe.
    df_dtypes = df.ts.types

    self._validate_compatible_datatypes(value, df, df_dtypes)

    # If replace_value is a scalar, then fillna with it.
    if np.isscalar(value):
      df = df.fillna(value, subset = df.columns)
    elif isinstance(value, dict):
      for col_name, value in value.items():
        # If the value is a list, then it should be an empty list.
        if isinstance(value, list):
          # If the value is an empty list, then replace null values 
          # with an empty array.
          df = df.withColumn(col_name, 
                             F.when(F.col(col_name).isNull(), F.array()
                                   ).otherwise(F.col(col_name))
                             )
        elif np.isscalar(value):
          df = df.fillna(value, subset = [col_name])

    elif isinstance(value, list):
      for col in df.columns:
        df = df.withColumn(col, 
                           F.when(F.col(col).isNull(), F.array()
                                 ).otherwise(F.col(col))
                           )

    return df

  def _validate_compatible_datatypes(self, value, df, df_dtypes):
      
    # Get the compatible datatypes of python and spark.
    datatype_dict = _get_compatible_datatypes_of_python_and_spark()
    
    if isinstance(value, dict):
      _ = self._validate_column_names(list(value.keys()))
    
      for col_name, col_value in value.items():
        col_dtype = type(col_value).__name__
        assert df_dtypes[col_name] in datatype_dict[col_dtype],\
          (f"replacement value for column '{col_name}' " 
            f"should be of type '{df_dtypes[col_name]}' "
          )
      
        if isinstance(col_value, list):
          assert len(col_value) == 0,\
        ("replacement value should be an empty list if a "
         "list is passed as a value"
        )

    elif isinstance(value, list):
      assert len(value) == 0,\
      ("replacement value should be an empty list "
       "if a list is passed as a value"
      )

      # Check if all the columns are of type array.
      for col in df.columns:
        assert df_dtypes[col] == "array",\
        ("replacement value should be an empty list "
         "if a list is passed as a value"
        )

    elif np.isscalar(value):
      value_dtype = type(value).__name__
      for col in df.columns:
        assert df_dtypes[col] in datatype_dict[value_dtype],\
        (f"replacement value for column '{col}' "
         f"should be of type {df_dtypes[col]}"
        )
        
    else:
      assert False, "value should be a scalar, an empty list or a dictionary"
