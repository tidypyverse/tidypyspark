tidypyspark development plan
----------------------------

attributes
----------
nrow, ncol, colnames, dim, shape

basic methods
--------------

- [] add_row_number
- [] add_group_number

- [x] select
- [x] arrange
- [x] distinct
- [x] mutate
- [ ] summarise (Sr)
- [ ] relocate (check if this physically moves data, soft signal: no) (Su)
- [ ] rename (Su)
- [ ] filter ( df.filter((lhs, ">", rhs), by = ) ) **pondering**



to methods
----------
- [ ] to_pandas (Su)

pipe methods
------------
- [ ] pipe (Sr)
- [ ] pipe_tee (Sr)

- [ ] glimpse (Sr)
    - take first few rows, convert to tidypandas and then call glimpse


Join methods (Ja)
------------
- [ ] join
- [ ] inner, outer/full, left, right
- [ ] semi, anti
- [ ] cross

Bind methods (Sr)
------------

- [ ] rbind, cbind

pivot methods (Ja)
-------------

- [ ] pivot_wider
- [ ] pivot_longer



Count methods (Sr)
-------------
- [ ] count
- [ ] add_count

slice methods
-------------

- [ ] slice_sample
- [ ] slice_min (Su)
- [ ] slice_max

Lower priority things
---------------------
Enhance _create_windowspec (Suyash)
Support group apply (Suyash)