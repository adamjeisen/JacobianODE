# Q3 / Q4 / Q5 findings


## Q3: When can we identify the eventual winner?


### lorenz_current

- max epoch in series: 111
- spearman ≥ 0.7 first reached at epoch 10 (9% of runtime)
- spearman ≥ 0.9 first reached at epoch 58 (52% of runtime)
```
 epoch    n   spearman  top3_overlap
    11   21      0.484             1/3
    27   21      0.608             1/3
    55   21      0.827             1/3
    83   19      0.921             2/3
    99   18      0.920             1/3
   111    7      0.929             1/3
```


### lorenz_current_obsnoise001

- max epoch in series: 113
- spearman ≥ 0.7 first reached at epoch 13 (12% of runtime)
- spearman ≥ 0.9 first reached at epoch 15 (13% of runtime)
```
 epoch    n   spearman  top3_overlap
    11   21      0.548             3/3
    28   21      0.925             2/3
    56   21      0.884             3/3
    84   21      0.906             1/3
   101   21      0.996             3/3
   113    5      1.000             1/3
```


### wmtask_vanilla_old

- max epoch in series: 11
- spearman ≥ 0.7 first reached at epoch 0 (0% of runtime)
- spearman ≥ 0.9 first reached at epoch 0 (0% of runtime)
```
 epoch    n   spearman  top3_overlap
     1    9      0.900             3/3
     2    9      0.933             3/3
     5    9      0.833             2/3
     8    7      1.000             3/3
     9    7      1.000             3/3
    11    6      1.000             2/3
```
