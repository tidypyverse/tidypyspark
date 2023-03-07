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
- [ ] filter ( df.filter((lhs, ">", rhs), by = ) ) **pondering**

to methods
----------
- [ ] to_pandas (deffered)

pipe methods
------------
- [x] pipe (Sr)
- [x] pipe_tee (Sr)
- [ ] glimpse (deffered)

Join methods (Ja)
------------
- [x] join
- [x] inner, outer/full, left, right
- [x] semi, anti
- [x] cross

Bind methods (Sr)
------------
- [ ] rbind, cbind

pivot methods (Ja)
-------------
- [ ] pivot_wider
- [ ] pivot_longer

Count methods (Sr)
-------------
- [x] count
- [x] add_count

slice methods
-------------
- [ ] slice_sample (Sr)
- [x] slice_min (Sr)
- [x] slice_max (Sr)

na methods
------------
- [ ] drop_na (Sr)
- [ ] replace_na (Sr)
- [ ] fill_na (Sr)

Lower priority things
---------------------
Enhance _create_windowspec (Suyash)
Support group apply (Suyash)