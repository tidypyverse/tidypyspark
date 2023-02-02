################################################################################
# Test tidypyspark code
################################################################################

import tidypyspark as ts

# start spark session ----
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.getOrCreate()

# test code ----
pen = spark.read.csv("pen.csv", header = True).drop("_c0")
pen.show(6)


pen.ts.add_row_number('bill_length_mm').filter().show(10)

(pen.ts.add_row_number('species', by = 'species')
    .filter(F.col('row_number') <= 2)
    .drop('row_number')
    .ts.add_group_number('species', name = 'gn')
    .show(10)
    )

# test: add_row_number
(md_chunk.ts.add_row_number(order_by = 'current_inventory',
                            by = 'system_item_nbr'
                            )
         .ts.select(['current_inventory', 'row_number', 'system_item_nbr'])
         .show(6)
         )
         
(md_chunk.ts.add_row_number(order_by = ('current_inventory', 'desc'))
         .ts.select(['current_inventory', 'row_number', 'system_item_nbr'])
         .show(20)
         )

# test: add_group_number         
(md_chunk.ts.add_group_number(by = 'system_item_nbr',
                              order_by = 'current_inventory'
                              )
         .ts.select(['current_inventory', 'group_number', 'system_item_nbr'])
         .show(20)
         )

(md_chunk.ts.select(['system_item_nbr'])
         .ts.add_row_number(order_by = 'system_item_nbr')
         .show()
         )

md_chunk.ts.add_row_number(order_by = [('md_start_date', 'desc')]).show(6)
# md_chunk.withColumn('rn', F.row_number().over(Window.orderby(col('A').desc())))

md_chunk.ts.add_row_number(order_by = 'md_start_date',
                           by = 'system_item_nbr'
                           ).show(6)
temp = (md_chunk.ts.add_group_number(by = 'system_item_nbr')
                .ts.select(['system_item_nbr', 'group_number', 'club_nbr'])
                )
temp.show(6)

md_chunk.ts.inner_join(temp, on = ['system_item_nbr', 'club_nbr']).show()

md_chunk._validate_by('retail_price')


(md_chunk.order_by('system_item_nbr').drop_duplicates(subset = ['oos_date']))

 
# end spark session ----
spark.stop()
