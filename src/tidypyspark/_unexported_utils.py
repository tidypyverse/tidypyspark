# ----------------------------------------------------------------------------
# This file is a part of tidypyspark python package
# Find the dev version here: https://github.com/talegari/tidypyspark
# ----------------------------------------------------------------------------
import string
import inspect
import warnings
import numpy as np

def _is_kwargable(func):
    res = False
    assert callable(func), "arg 'func' should be callable"
    try:
        spec = inspect.getfullargspec(func)
        if spec.varkw is not None:
            res = True
    except TypeError as e:
        pass # res is False
    return res

def _is_valid_colname(string):
    res = (isinstance(string, str)) and (len(string) != 0) and (string[0] != "_")
    return res
  
def _is_string_or_string_list(x):
    '''
    _is_string_or_string_list(x)
    
    Check whether the input is a string or a list of strings

    Parameters
    ----------
    x : object
        Any python object

    Returns
    -------
    bool
    True if input is a string or a list of strings
    
    Examples
    --------
    >>> _is_string_or_string_list("bar")      # True
    >>> _is_string_or_string_list(["bar"])    # True
    >>> _is_string_or_string_list(("bar",))   # False
    >>> _is_string_or_string_list(["bar", 1]) # False
    '''
    res = False
    if isinstance(x, str):
        res = True
    elif isinstance(x, list) and len(x) >= 1:
        if all([isinstance(i, str) for i in x]):
            res = True
    else:
        res = False
    
    return res
    
def _enlist(x):
    '''
    _enlist(x)
    
    Returns the input in a list (as first element of the list) unless input itself is a list

    Parameters
    ----------
    x : object
        Any python object

    Returns
    -------
    list
    Returns the input in a list (as first element of the list) unless input itself is a list
    
    Examples
    --------
    >>> _enlist(["a"]) # ["a"]
    >>> _enlist("a")   # ["a"]
    >>> _enlist((1, )) # [(1, )]
    '''
    if not isinstance(x, list):
        x = [x]
    
    return x

def _get_unique_names(strings):
    '''
    _get_unique_names(strings)
    
    Returns a list of same length as the input such that elements are unique. This is done by adding '_1'. The resulting list does not alter nth element if the nth element occurs for the first time in the input list starting from left.
    
    Parameters
    ----------
    strings : list
        A list of strings

    Returns
    -------
    list of strings
    
    Examples
    --------
    >>> _get_unique_names(['a', 'b'])               # ['a', 'b']
    >>> _get_unique_names(['a', 'a'])               # ['a', 'a_1']
    >>> _get_unique_names(['a', 'a', 'a_1'])        # ['a', 'a_1_1', 'a_1']
    '''
    assert _is_string_or_string_list(strings)
    strings = _enlist(strings)

    new_list = []
    old_set = set(strings)
    
    for astring in strings:
        counter = 0 
        while True:
            if astring in new_list:
                counter = 1
                astring = astring + "_1" 
            elif astring in old_set:
                if counter > 0:
                    astring = astring + "_1"
                else:
                    new_list.append(astring)
                    try:
                        old_set.remove(astring)
                    except:
                        pass
                    break
            else:
                new_list.append(astring)
                try:
                    old_set.remove(astring)
                except:
                    pass
                break
        
    return new_list

def _is_unique_list(x):
    '''
    _is_unique_list(x)
    
    Returns True if input list does not have duplicates

    Parameters
    ----------
    x : list

    Returns
    -------
    bool
    '''
    assert isinstance(x, list)
    return len(set(x)) == len(x)

def _generate_new_string(strings):
    
    assert isinstance(strings, list)
    assert all([isinstance(x, str) for x in strings])
    
    while True:
        random_string = "".join(np.random.choice(list(string.ascii_letters), 20))
        if random_string not in strings:
            break
    
    return random_string

def _is_nested(x):
    assert isinstance(x, (list, tuple, set)) or np.isscalar(x)
    
    if np.isscalar(x):
        res = False
    else:
        res = not all([np.isscalar(y) for y in x])
    
    return res

def _flatten_strings(x):
    res = set()
    x = list(x)
    for ele in x:
        if isinstance(ele, str):
            res = res.union({ele})
        elif _is_nested(ele):
            res = res.union(_flatten_strings(ele))
        else:
            res = res.union(set(ele))
    return list(res)

def _nested_is_unique(x):
    res = list()
    x = list(x)
    for ele in x:
        if isinstance(ele, str):
            res = res + [ele]
        elif _is_nested(ele):
            res = res + _flatten_strings(ele)
        else:
            assert _is_string_or_string_list(list(ele)),\
                "Each element of the nested structure should be a string"
            res = res + list(ele)
    return (len(res) == len(set(res)))

def _is_perfect_sublist(subset_list, full_list):
    if subset_list[0] in full_list:
        start_index = full_list.index(subset_list[0])
        for i in range(len(subset_list)):
            if full_list[start_index+i] != subset_list[i]:
                return False
        return True
    return False

def _get_compatible_datatypes_of_python_and_spark():

    '''
    get_compatible_datatypes_of_python_and_spark()
    
    Returns a dictionary of data types that are compatible with both python
    and spark. The keys are the python data types and the values are the spark
    data types.
    
    Returns
    -------
    dict<string, set>
    '''
    return {
        "str": set(["string"]),
        "int": set(["integer", "long"]),
        "float": set(["double", "float"]),
        "bool": set(["boolean"]),
        "list": set(["array"])
    }
