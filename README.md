[![PyPI
version](https://badge.fury.io/py/tidypyspark.svg)](https://badge.fury.io/py/tidypyspark)

# `tidypyspark`

> Make [pyspark](https://pypi.org/project/pyspark/) sing
> [dplyr](https://dplyr.tidyverse.org/)

> Inspired by [sparklyr](https://spark.rstudio.com/),
> [tidyverse](https://tidyverse.tidyverse.org/)

`tidypyspark` python package provides *minimal, pythonic* wrapper around
pyspark sql dataframe API in
[tidyverse](https://tidyverse.tidyverse.org/) flavor.

-   With accessor `ts`, apply `tidypyspark` methods where both input and
    output are mostly pyspark dataframes.
-   Consistent 'verbs' (`select`, `arrange`, `distinct`, ...)

Also see [`tidypandas`](https://pypi.org/project/tidypandas/): A
**grammar of data manipulation** for
[pandas](https://pandas.pydata.org/docs/index.html) inspired by
[tidyverse](https://tidyverse.tidyverse.org/)

## Usage

    # assumed that pyspark session is active
    from tidypyspark import ts 
    import pyspark.sql.functions as F
    from tidypyspark.datasets import get_penguins_path

    pen = spark.read.csv(get_penguins_path(), header = True, inferSchema = True)

    (pen.ts.add_row_number(order_by = 'bill_depth_mm')
        .ts.mutate({'cumsum_bl': F.sum('bill_length_mm')},
                   by = 'species',
                   order_by = ['bill_depth_mm', 'row_number'],
                   range_between = (-float('inf'), 0)
                   )
        .ts.select(['species', 'bill_length_mm', 'cumsum_bl'])
        ).show(5)
        
    +--------------+-------+-------------+------------------+
    |bill_length_mm|species|bill_depth_mm|         cumsum_bl|
    +--------------+-------+-------------+------------------+
    |          32.1| Adelie|         15.5|              32.1|
    |          35.2| Adelie|         15.9| 67.30000000000001|
    |          37.7| Adelie|           16|105.00000000000001|
    |          36.2| Adelie|         16.1|141.20000000000002|
    |          33.1| Adelie|         16.1|             174.3|
    +--------------+-------+-------------+------------------+

## Example

-   `tidypyspark` code:

<!-- -->

    (pen.ts.select(['species','bill_length_mm','bill_depth_mm', 'flipper_length_mm'])
     .ts.pivot_longer('species', include = False)
     ).show(5)
     
     +-------+-----------------+-----+
    |species|             name|value|
    +-------+-----------------+-----+
    | Adelie|   bill_length_mm| 39.1|
    | Adelie|    bill_depth_mm| 18.7|
    | Adelie|flipper_length_mm|  181|
    | Adelie|   bill_length_mm| 39.5|
    | Adelie|    bill_depth_mm| 17.4|
    +-------+-----------------+-----+

-   equivalent pyspark code:

<!-- -->

    stack_expr = '''
                 stack(3, 'bill_length_mm', `bill_length_mm`,
                          'bill_depth_mm', `bill_depth_mm`,
                          'flipper_length_mm', `flipper_length_mm`)
                          as (`name`, `value`)
                 '''
    pen.select('species', F.expr(stack_expr)).show(5)

> `tidypyspark` relies on the amazing `pyspark` library and spark
> ecosystem.

## Installation

`pip install tidypyspark`

-   On github: <https://github.com/talegari/tidypyspark>
-   On pypi: <https://pypi.org/project/tidypyspark>
