# ----------------------------------------------------------------------------
# This file is a part of tidypyspark python package
# Find the dev version here: https://github.com/talegari/tidypyspark
# ---------------------------------------------------------------------
import warnings
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import re
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
        allowed = ['asc', 'desc', 'asc_null_first', 'desc_nulls_first']
        if x[1] not in allowed:
          raise Exception((f'Second element of the input tuple should '
                           f'one among: {allowed}. '
                           f'Input: {x}'
                           ))
        
        assert x[0] in cns,\
          (f'String input to order_by should be a valid column name. '
           f'Input: {x[0]}'
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
          (f'String input to order_by should be a valid column name. '
           f'Input: {x}'
           )
        order_by[id] = F.col(x).asc_nulls_last()
      
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
        allowed = ['asc', 'desc', 'asc_null_first', 'desc_nulls_first']
        if x[1] not in allowed:
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

  def rename(self, old_new_dict):
    '''
    rename
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
    pen.ts.rename({'species': 'species_2', "year":"year_2})

    Invalid Example
    pen.ts.rename({'species': 'species_2', "species_2":"species_3})
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
    assert len(old_new_dict)>0, \
        "there should alteast one element in old_new_dict"

    # new names should not intersect with 'remaining' names
    remaining = set(cn).difference(old_new_dict.keys())
    assert len(remaining.intersection(old_new_dict.values())) == 0, \
      ("New intended column names (values of the dict) lead to duplicate "
       "column names"
       )
    select_col_with_alias = [F.col(c).alias(old_new_dict.get(c, c)) for c in self.colnames]
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
    column before which the column are to be moved. The default is None.
    after : string, optional
    column after which the column are to be moved. The default is None.

    Returns
    -------
    pyspark dataframe

    Notes
    -----
    Only one among 'before' and 'after' can be not None. When both are None,
    the columns are added to the beginning of the dataframe (leftmost)

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
    
  # join methods --------------------------------------------------------------------
  def _validate_join(self, pyspark_df, on, on_x, on_y , sql_on, suffix, how):
      
    assert isinstance(pyspark_df, pyspark.sql.dataframe.DataFrame),\
      "'pyspark_df' should be a pyspark dataframe"
    
    assert isinstance(how, str),\
      "arg 'how' should be a string"
    assert how in ['inner', 'left', 'right', 'full', 'semi', 'anti'],\
      "arg 'how' should be one among: 'inner', 'left', 'right', 'full', 'semi', 'anti', 'cross'"
    
    cn_x = self.colnames
    print(cn_x)
    cn_y = pyspark_df.columns
    print(cn_y)
  
    assert ((on is not None) + ((on_x is not None) and (on_y is not None)) 
            + (sql_on is not None) == 1),\
      "Exactly one among 'on', 'sql_on', 'on_x and on_y' should be specified"
  
    if on is not None:
      assert _is_string_or_string_list(on),\
        ("'on' should be a string or a list of strings of common "
          "column names"
        )
      on = _enlist(on)
      assert _is_unique_list(on),\
        "arg 'on' should not have duplicates"
      assert set(on).issubset(cn_x),\
        "arg 'on' should be a subset of column names of x"
      assert set(on).issubset(cn_y),\
        "arg 'on' should be a subset of column names of y"
    elif sql_on is not None:
      assert isinstance(sql_on, str),\
        "arg 'sql_on' should be a string"
    else:
      assert on_x is not None and on_y is not None,\
        ("When arg 'on' is None, " 
          "both args'on_x' and 'on_y' should not be None"
        )
      assert _is_string_or_string_list(on_x),\
        "arg 'on_x' should be a string or a list of strings"
      assert _is_string_or_string_list(on_y),\
        "arg 'on_y' should be a string or a list of strings"
      
      on_x = _enlist(on_x)
      on_y = _enlist(on_y)
      
      assert _is_unique_list(on_x),\
        "arg 'on_x' should not have duplicates"
      assert _is_unique_list(on_y),\
        "arg 'on_y' should not have duplicates"    
      assert set(on_x).issubset(cn_x),\
        "arg 'on_x' should be a subset of column names of x"
      assert set(on_y).issubset(cn_y),\
        "arg 'on_y' should be a subset of column names of y"
      assert len(on_x) == len(on_y),\
        "Lengths of arg 'on_x' and arg 'on_y' should match"
      
    assert isinstance(suffix, list),\
      "arg 'suffix' should be a list"
          
    assert len(suffix) == 2,\
      "arg 'suffix' should be a list of length 2"
        
    assert isinstance(suffix[0], str) and isinstance(suffix[1], str),\
      "arg 'suffix' should be a list of two strings"
    
    assert suffix[0] != suffix[1],\
      "left and right suffix should be different."
    
    return None

  def _execute_on_command_for_join(self, on, suffix, how, LHS, RHS):
          
    on = _enlist(on)
    cols_LHS = LHS.columns
    cols_RHS = RHS.columns

    # Create a dictionary with old column names as keys and values as old column names + suffix
    new_cols_LHS = list(setlist(cols_LHS) - setlist(on))
    old_new_dict_LHS = {col: col + suffix[0] for col in new_cols_LHS}      

    # Create a dictionary with old column names as keys and values as old column names + suffix
    new_cols_RHS = list(setlist(cols_RHS) - setlist(on))
    old_new_dict_RHS = {col: col + suffix[1] for col in new_cols_RHS}

    assert len(list(set(old_new_dict_LHS.values()).intersection(old_new_dict_RHS.values()))) == 0,\
      "Column names should be unique after joining the dataframes"
    
    # Create a list of columns with alias for LHS df in pyspark convention.
    select_col_with_alias_LHS = [F.col(c).alias(old_new_dict_LHS.get(c, c)) for c in cols_LHS]
    # Create the new LHS df with the new column names.
    new_LHS = LHS.select(select_col_with_alias_LHS)
    
    # Create a list of columns with alias for RHS df in pyspark convention.
    select_col_with_alias_RHS = [F.col(c).alias(old_new_dict_RHS.get(c, c)) for c in cols_RHS]
    # Create the new RHS df with the new column names.
    new_RHS = RHS.select(select_col_with_alias_RHS)

    # Join the dataframes
    res = new_LHS.join(new_RHS, on = on, how = how)

    # Get the count of columns in the original joined dataframe
    count_cols = Counter(cols_LHS + cols_RHS)
    
    renamed_col_dict = {} # Dictionary to store the renamed columns
    for col in res.columns: 
      for s in suffix:  # use a loop to check each suffix in the list
        if len(s) > 0 and col.endswith(s):  # check if the column name ends with the suffix
            new_col = col[:-len(s)] # remove the suffix from the column name
            # Check if the new column name is not a duplicate and is not in the list of columns to be joined
            if count_cols[new_col] < 2 and new_col not in on:      
                renamed_col_dict[col] = new_col 

    # Rename the columns in the dataframe
    if len(renamed_col_dict) > 0:
      res = res.ts.rename(renamed_col_dict)

    return res
  
  def _execute_sql_on_command_for_join(self, sql_on, suffix, how, LHS, RHS):

    cols_LHS = LHS.columns
    cols_RHS = RHS.columns

    # Create a dictionary with old column names as keys and values as old column names + suffix
    old_new_dict_LHS = {col: col + suffix[0] for col in cols_LHS}
    old_new_dict_RHS = {col: col + suffix[1] for col in cols_RHS}

    assert len(list(set(old_new_dict_LHS.values()).intersection(old_new_dict_RHS.values()))) == 0,\
      "Column names should be unique after joining the dataframes" 

    # Get the column names from the sql_on command.
    # e.g Format - {'LHS': ['dept', 'id'], 'RHS': ['dept', 'id', 'age']}
    column_names_tuple_list = self._extract_cols_from_sql_on_command(sql_on)
    
    # Get the sql_on statement with suffix.
    # e.g Format - "LHS.dept = RHS.dept_y and LHS.id = RHS.id_y"
    sql_on = self._get_sql_on_statement_with_suffix(sql_on, suffix, column_names_tuple_list)

    # Rename the columns of the LHS and RHS dataframes.
    LHS = LHS.ts.rename(old_new_dict_LHS) 
    RHS = RHS.ts.rename(old_new_dict_RHS)

    # Join the dataframes
    res = LHS.join(RHS, on = eval(sql_on), how = how)

    # Get the count of columns in the original joined dataframe
    res = self._get_spark_df_by_removing_suffix(suffix, cols_LHS, cols_RHS, res)

    return res
  
  def _execute_on_x_on_y_command_for_join(self, on_x, on_y, suffix, how, LHS, RHS):

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

  def _get_sql_on_statement_with_suffix(self, sql_on, suffix, column_names_tuple_list):
      
    # Create a list of column names with suffix for LHS and RHS.
    # e.g Format - ['LHS.dept', 'LHS.id', 'RHS.dept', 'RHS.id', 'RHS.age']
    sql_on_LHS_cols = ['LHS.' + col for col in column_names_tuple_list['LHS']]
    sql_on_RHS_cols = ['RHS.' + col for col in column_names_tuple_list['RHS']]

    # Create a dictionary with old column names as keys and values as old column names + suffi
    # e.g Format - {'LHS.dept': 'new_LHS.dept_suffix[0]', 
    #               'RHS.dept': 'new_RHS.dept_suffix[1]', 'RHS.age': 'new_RHS.age_suffix[1]'}
    sql_on_LHS_cols_dict = {col: col + suffix[0] for col in sql_on_LHS_cols}
    sql_on_RHS_cols_dict = {col: col + suffix[1] for col in sql_on_RHS_cols}
    # Merge the two dictionaries
    sql_on_cols_dict = {**sql_on_LHS_cols_dict, **sql_on_RHS_cols_dict}
    
    # Replace the column names in the sql_on command with the new column names in the sql_on_cols_dict
    for key, value in sql_on_cols_dict.items():
      sql_on = sql_on.replace(key, value)
    
    return sql_on
  
  def _extract_cols_from_sql_on_command(self, sql_on):

    # Set regex pattern.
    # e.g. 'LHS.dept == RHS.dept' will be converted to [('LHS', 'dept'), ('RHS', 'dept')]
    pattern = '([a-zA-Z0-9]+)\.([a-zA-Z0-9]+)'

    # Get all column names with their table names
    # Format - [('LH', 'dept'), ('LHS', 'dept'), ('RHS', 'age')]
    column_names_tuple_list = re.findall(pattern, sql_on)

    # Filtering tuples having only LHS or RHS as the first element of the tuple
    filtered_tuples = [(key, value) for (key, value) in column_names_tuple_list if key=='LHS' or key=='RHS']

    # Generates a dictionary with LHS and RHS as keys and column names as values.
    # e.g. {'RHS': ['id2', 'dept', 'age'], 'LHS': ['dept']}
    dict_from_tuple = {key:[] for (key, _) in filtered_tuples}
    for tpl in filtered_tuples:
        dict_from_tuple[tpl[0]].append(tpl[1])

    return dict_from_tuple
  
  def _get_spark_df_by_removing_suffix(self, suffix, cols_LHS, cols_RHS, res):
      
    # Get the frequency count of columns in the original joined dataframe.
    count_cols = Counter(cols_LHS + cols_RHS)
  
    renamed_col_dict = {} # Dictionary to store the renamed columns
    for col in res.columns: 
      for s in suffix:  # use a loop to check each suffix in the list
        if len(s) > 0 and col.endswith(s):  # check if the column name ends with the suffix
            new_col = col[:-len(s)] # remove the suffix from the column name
            if count_cols[new_col] < 2: # Check if the new column name is not a duplicate    
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

    # Create a dictionary with old column names as keys and values as old column names + suffix
    old_new_dict_LHS = {col: col + suffix[0] for col in cols_LHS}
    old_new_dict_RHS = {col: col + suffix[1] for col in cols_RHS}

    assert len(set(old_new_dict_LHS.values()).intersection(old_new_dict_RHS.values())) == 0,\
      "Column names should be unique after joining the dataframes" 
    
    # Rename the columns in the LHS and RHS dataframes
    LHS = LHS.ts.rename(old_new_dict_LHS)
    RHS = RHS.ts.rename(old_new_dict_RHS)
  
    # Perform cross join.
    res = LHS.crossJoin(RHS)

    # Remove the unnecessary suffix(es) from the column names
    res = self._get_spark_df_by_removing_suffix(suffix, cols_LHS, cols_RHS, res)

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
    Joins columns of y to self by performing different types of joins.
    
    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge
    how: string
        Type of join to be performed. Default is 'inner'.
        Other options are 'left', 'right', 'outer', 'full', 'cross', 'semi', 'anti'
      
    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)

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

    >>> df1.ts.inner_join(df2, 
    >>>        sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)', how = 'inner).show()
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
      res = self._execute_sql_on_command_for_join(sql_on, suffix, how, LHS, RHS)
    else:
      res = self._execute_on_x_on_y_command_for_join(on_x, on_y, suffix, how, LHS, RHS)

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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge
      
    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)

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

    >>> df1.ts.left_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings 
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge

    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)
    
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

    >>> df1.ts.right_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings 
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge

    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)
    
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

    >>> df1.ts.inner_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings 
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge

    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)
    
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

    >>> df1.ts.full_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge
      
    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)

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

    >>> df1.ts.anti_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
    
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix = ["", "_y"], how = 'anti')
  
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
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    on: string or a list of strings
        Common column names to match
    on_x: string or a list of strings
        Column names of self to be matched with arg 'on_y'
    on_y: string or a list of strings
        Column names of y to be matched with arg 'on_x'
    sql_on: string
        SQL expression used to join both DataFrames. Recommended for inequality joins.
        The left table has to be specified as 'LHS' and the right table as 'RHS'.
        e.g. '(LHS.dept == RHS.dept) & (LHS.age == RHS.age) & (RHS.age < 30)'
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge
      
    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)

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

    >>> df1.ts.semi_join(df2, sql_on = '(LHS.id == RHS.id) & (LHS.dept == RHS.dept) & (RHS.age < 30)').show()
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
     
    return self.join(pyspark_df, on, on_x, on_y, sql_on, suffix = ["", "_y"], how = 'semi')
  
  def cross_join(self, pyspark_df, suffix = ["", "_y"]):
    '''
    Returns the cartesian product of two DataFrames.

    Parameters
    ----------
    pyspark_df (pyspark.sql.DataFrame): DataFrame to join with current DataFrame.
    suffix: list of two stings
        suffix to append the columns of left and right in order to create unique names after the merge
      
    Returns
    -------
    joined DataFrame (pyspark.sql.DataFrame)

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
    cn = self._clean_column_names(column_names)
    
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
    pen.ts.add_count(['species', 'sex']).show()
    pen.ts.add_count('species', name = 'cnt').show()
    pen.ts.add_count('species', wt = 'body_mass_g').show()
    '''
    cn = self._clean_column_names(column_names)
    
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
    pen.ts.slice_min(n = 2,
                     order_by_column = 'bill_depth_mm',
                     with_ties = False,
                     by = ['species', 'sex']
                     )
    '''
    
    assert isinstance(n, int) and n > 0,\
      "n should be a positive integer"
    order_by_spec = self._clean_order_by((order_by_column, 'asc'))
    assert isinstance(with_ties, bool)
    
    # create windowspec
    if by is None:
      win = self._create_windowspec(order_by = order_by_spec)
    else:
      by = self._clean_by(by)
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
      Name of the column to order by in descending nulls to last
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
    pen.ts.slice_max(n = 2,
                     order_by_column = 'bill_depth_mm',
                     with_ties = False,
                     by = ['species', 'sex']
                     )
    '''
    assert isinstance(n, int) and n > 0,\
      "n should be a positive integer"
    order_by_spec = self._clean_order_by((order_by_column, 'desc'))
    assert isinstance(with_ties, bool)
    
    # create windowspec
    if by is None:
      win = self._create_windowspec(order_by = order_by_spec)
    else:
      by = self._clean_by(by)
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
    