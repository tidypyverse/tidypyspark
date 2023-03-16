tidypyspark development plan
----------------------------

attributes
----------
nrow, ncol, colnames, dim, shape

basic methods
--------------
- [x] add_row_number
- [x] add_group_number
- [x] select
- [x] arrange
- [x] distinct
- [x] mutate
- [x] summarise
- [x] relocate
- [x] rename
- [x] filter

to_methods
----------
- [x] pull (to_series)
- [x] to_list
- [x] to_dict
- [x] to_pandas

show_methods
------------
- [ ] glimpse (Su)

pipe methods
------------
- [x] pipe
- [x] pipe_tee 

Join methods
------------
- [x] join
- [x] inner, outer/full, left, right
- [x] semi, anti
- [x] cross

Bind methods
------------
- [x] rbind
- [x] union

pivot methods
-------------
- [x] pivot_wider
- [x] pivot_longer

Count methods
-------------
- [x] count
- [x] add_count

slice methods
-------------
- [ ] slice_sample (Sr)
- [x] slice_min
- [x] slice_max

na methods
------------
- [ ] drop_na (Ja)
- [ ] replace_na (Ja)
- [x] fill_na

nest methods
------------
- [x] nest
- [x] unnest
- [x] unnest_wider
- [ ] unnest_longer (Sr)

Lower priority things
---------------------
Enhance _create_windowspec (Suyash)
Support group apply (Suyash)