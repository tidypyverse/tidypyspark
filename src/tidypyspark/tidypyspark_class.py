import warnings
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from accessor_class import register_dataframe_accessor
from _unexported_utils import (
                              _is_kwargable,
                              _is_valid_colname,
                              _is_string_or_string_list,
                              _enlist,
                              _get_unique_names,
                              _is_unique_list,
                              _get_dtype_dict,
                              _generate_new_string,
                              _coerce_series,
                              _coerce_pdf,
                              _is_nested,
                              _flatten_strings,
                              _nested_is_unique
                            )

# tidypyspark ----
@register_dataframe_accessor('ts')
class acc_on_pyspark():
  def __init__(self, data):
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
  # A cleaner returns the cleaned object
  
  def _clean_by(self, by):
    '''
    _clean_by
    cleans 'by' and returns a list of 'col's
    
    Parameters
    ----------
    by : string, list of strings, list of 'col's

    Returns
    -------
    by2 : list
        list of 'col's
    
    Examples (prototypes)
    --------
    'a' --> F.col('a')
    ['a', 'b'] --> [F.col('a'), F.col('b')]
    ['a', F.col('c')] --> [F.col('a'), F.col('c')]
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
    Parameters
    ----------
    order_by : TYPE
        DESCRIPTION.

    Raises
    ------
    exception
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

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
          raise Exception(f'Second element of the input tuple should one among: ["asc", "desc"]. '
           f'Input: {x}'
           )
        
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
        raise Exception("An element of 'order_by' should be a tuple or a string")
      
    return order_by
      
  def _clean_column_names(self, column_names):
    
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
          raise Exception(f'Second element of the input tuple should one among: ["asc", "desc"]. '
           f'Input: {x}'
           )
        
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
        raise Exception("An element of 'order_by' should be a tuple or a string")
      
    return out
    
  def _create_windowspec(self, **kwargs):

    if 'by' in kwargs:
      win = Window.partitionBy(kwargs['by'])
    
    if 'order_by' in kwargs:
      if 'win' in locals():
        win = win.orderBy(kwargs['order_by'])
      else:
        win = Window.orderBy(kwargs['order_by'])
            
    if 'range_between' in kwargs:
      win = win.rangeBetween(**kwargs['range_between'])
    
    if 'rows_between' in kwargs:
      win = win.rowsBetween(**kwargs['rows_between'])
    
    return win
  
  # utils -------------------------------------------------------------------
  def add_row_number(self, order_by, name = "row_number", by = None):
    order_by = self._clean_order_by(order_by)
    
    if by is not None:
        by = self._clean_by(by)
        win = self._create_windowspec(by = by, order_by = order_by)
    else:
        win = self._create_windowspec(order_by = order_by)
    
    res = self.__data.withColumn(name, F.row_number().over(win))
    return res
  
  def add_group_number(self, by, order_by = None, name = "group_number"):
      
      by = self._clean_by(by)
      order_by_cols = self._extract_order_by_cols(order_by)
      
      if order_by is None:
          order_by = self._clean_order_by(by)
      else:
          order_by = self._clean_order_by(order_by)
          
      win = self._create_windowspec(order_by = order_by)
      
      res = (self.__data
                 .select(list(set(by).union(order_by_cols)))
                 .dropDuplicates(subset = by)
                 .withColumn(name, F.row_number().over(win))
                 )
      
      res = (self.__data.join(res, how = "inner", on = by))       
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
      pyspark.sql.dataframe.DataFrame
      
      Examples
      --------
      
      '''
      cn = self._clean_column_names(column_names)
      
      if not include:
          cn = list(set(self.colnames).difference(cn))
          
      res = self.__data.select(*cn)
      return res
  
  def arrange(self, order_by):
      order_by = self._clean_order_by(order_by)
      warnings.warn(
          "1. 'arrange' is not memory efficient as it brings all data "
           "to a single node"
           )
      warnings.warn(
          "2. 'arrange' is have no effect on subsequent operations, use "
           "'order_by' argument in required method"
           )
      return res.order_by(order_by)
      
  def distinct(self, column_names = None, order_by = None, keep_all = False):
      
      if column_names is None:
        column_names = self.colnames
      
      cn = self._clean_column_names(column_names)
      
      if order_by is None:
          res = self.__data.dropDuplicates(cn)
      else:
          order_by = self._clean_order_by(order_by)
          win = self._create_windowspec(order_by = order_by)
          rank_colname = _generate_new_name(self.colnames)
          
          res = (self.__data
                     .withColumn(rank_colname, F.row_number().over(win))
                     .dropDuplicates(cn + [rank_colname])
                     .drop(rank_colname)
                     )
          
      if not keep_all:
          res = res.select(*cn)
      
      return res
  
  def count(self, column_names, name = 'n'):
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
  
  def mutate(self, dictionary, window_spec = None, **kwargs):
      
      res = self.__data
      
      with_win = False
      # windowspec gets first preference
      if window_spec is not None:
          assert isinstance(w, pyspark.sql.window.WindowSpec),\
              ("'window_spec' should be an instance of"
               "'pyspark.sql.window.WindowSpec' class"
               )
          if len(kwargs) >= 1:
              print("'window_spec' takes precedence over other kwargs"
                    " in mutate"
                    )
          with_win = True
      else:
          # create windowspec if required
          if len(kwargs) >= 1:
              win = self._create_windowspec(**kwargs)
              with_win = True
          
      if with_win:
          for key, value in dictionary.items():
              res = res.withColumn(key, value)
          else:
              res = res.withColumn(key, (value).over(win))
              
      return res
  
  def join(self, pyspark_df, on = None, sql_on = None):
      
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
