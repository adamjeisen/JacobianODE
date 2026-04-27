# Q5 / Q6: sweep dominance, warmup, ES, alpha traj


#### lorenz_current — early-stopping config

```
  training.early_stopping.early_stopping_patience: [np.int64(5)]
  training.early_stopping.early_stopping_mode: ['percent_thresh']
  training.early_stopping.percent_thresh: [np.float64(0.01)]
  training.trainer_params.max_epochs: [np.int64(200)]
  training.trainer_params.max_steps: []
  training.trainer_params.limit_train_batches: [np.int64(200)]
  training.trainer_params.accumulate_grad_batches: [np.int64(1)]
```


#### lorenz_current — encoder/dynamics warmup config

encoder_warmup_epochs unique values in cfg: []

dynamics_warmup_epochs unique values in cfg: []


#### lorenz_current — sweep axis training.lightning.loop_closure_weight

```
 axis_val  final_best  final_runtime_s  data.flow.random_state
 0.000000    0.005649     10548.547667                      42
 0.000000    0.006217     10644.766559                      42
 0.000000    0.005166     10631.649470                      42
 0.000001    0.005496     10567.507704                      42
 0.000001    0.006016     10599.090254                      42
 0.000001    0.005354     10620.118996                      42
 0.000010    0.005572     10614.170803                      42
 0.000010    0.006184     10611.445556                      42
 0.000010    0.005620     10602.172315                      42
 0.000100    0.007447      5309.064833                      42
 0.000100    0.006483     10583.906785                      42
 0.000100    0.005526     10620.144510                      42
 0.001000    0.005618     10576.817845                      42
 0.001000    0.007271     10635.234359                      42
 0.001000    0.005961     10637.118750                      42
 0.010000    0.006089     10578.816019                      42
 0.010000    0.010159      9357.001731                      42
 0.010000    0.007817     10632.770029                      42
 0.100000    0.007550     10657.658480                      42
 0.100000    0.009500     10593.799749                      42
 0.100000    0.007540     10567.693974                      42
```


**Final best val by training.lightning.loop_closure_weight:**

```
          count   median      min      max
axis_val                                  
0.000000      3  0.00565  0.00517  0.00622
0.000001      3  0.00550  0.00535  0.00602
0.000010      3  0.00562  0.00557  0.00618
0.000100      3  0.00648  0.00553  0.00745
0.001000      3  0.00596  0.00562  0.00727
0.010000      3  0.00782  0.00609  0.01016
0.100000      3  0.00755  0.00754  0.00950
```


**Within-axis-bin std of final_best (median across bins):** 0.00087

**Between-axis-bin std of bin-medians:** 0.00096

**Ratio between/within:** 1.10× (sweep marginal)


#### lorenz_current — alpha_teacher_forcing trajectory

n_runs=21, alpha_start median=0.971, alpha_final median=0.0489, alpha_min median=0.0489, epoch_at_min median=108


#### lorenz_current_obsnoise001 — early-stopping config

```
  training.early_stopping.early_stopping_patience: [np.int64(5)]
  training.early_stopping.early_stopping_mode: ['percent_thresh']
  training.early_stopping.percent_thresh: [np.float64(0.01)]
  training.trainer_params.max_epochs: [np.int64(200)]
  training.trainer_params.max_steps: []
  training.trainer_params.limit_train_batches: [np.int64(200)]
  training.trainer_params.accumulate_grad_batches: [np.int64(1)]
```


#### lorenz_current_obsnoise001 — encoder/dynamics warmup config

encoder_warmup_epochs unique values in cfg: []

dynamics_warmup_epochs unique values in cfg: []


#### lorenz_current_obsnoise001 — sweep axis training.lightning.loop_closure_weight

```
 axis_val  final_best  final_runtime_s  data.flow.random_state
 0.000000    0.000577     10610.873231                      42
 0.000000    0.000590     10638.663827                      42
 0.000000    0.000784     10582.933914                      42
 0.000001    0.000581     10556.286330                      42
 0.000001    0.000666     10634.051909                      42
 0.000001    0.000794     10563.485603                      42
 0.000010    0.000582     10627.419767                      42
 0.000010    0.000661     10596.097791                      42
 0.000010    0.000828     10602.027537                      42
 0.000100    0.000629     10605.976930                      42
 0.000100    0.000584     10643.237613                      42
 0.000100    0.000884     10589.872556                      42
 0.001000    0.000799     10639.115951                      42
 0.001000    0.000690     10557.437478                      42
 0.001000    0.000856     10588.212316                      42
 0.010000    0.001171     10598.996116                      42
 0.010000    0.001135     10615.947713                      42
 0.010000    0.001283     10596.105257                      42
 0.100000    0.002786     10559.460573                      42
 0.100000    0.001686     10606.464854                      42
 0.100000    0.002214     10577.914811                      42
```


**Final best val by training.lightning.loop_closure_weight:**

```
          count   median      min      max
axis_val                                  
0.000000      3  0.00059  0.00058  0.00078
0.000001      3  0.00067  0.00058  0.00079
0.000010      3  0.00066  0.00058  0.00083
0.000100      3  0.00063  0.00058  0.00088
0.001000      3  0.00080  0.00069  0.00086
0.010000      3  0.00117  0.00113  0.00128
0.100000      3  0.00221  0.00169  0.00279
```


**Within-axis-bin std of final_best (median across bins):** 0.00012

**Between-axis-bin std of bin-medians:** 0.00059

**Ratio between/within:** 5.06× (sweep DOES matter)


#### lorenz_current_obsnoise001 — alpha_teacher_forcing trajectory

n_runs=21, alpha_start median=0.972, alpha_final median=0.0681, alpha_min median=0.0681, epoch_at_min median=104


#### wmtask_vanilla_old — early-stopping config

```
  training.early_stopping.early_stopping_patience: [np.int64(2)]
  training.early_stopping.early_stopping_mode: ['percent_thresh']
  training.early_stopping.percent_thresh: [np.float64(0.01)]
  training.trainer_params.max_epochs: [np.int64(1000)]
  training.trainer_params.max_steps: []
  training.trainer_params.limit_train_batches: [np.int64(500)]
  training.trainer_params.accumulate_grad_batches: [np.int64(4)]
```


#### wmtask_vanilla_old — encoder/dynamics warmup config

encoder_warmup_epochs unique values in cfg: []

dynamics_warmup_epochs unique values in cfg: []


#### wmtask_vanilla_old — sweep axis training.lightning.loop_closure_weight

```
 axis_val  final_best  final_runtime_s  data.flow.random_state
 0.000000    0.010179      3785.564007                      42
 0.000001    0.009711      2463.054886                      42
 0.000010    0.009742      2564.904805                      42
 0.000100    0.012840      2596.173787                      42
 0.001000    0.024156      1719.164809                      42
 0.010000    0.039941      1711.417001                      42
 0.100000    0.017704      5012.708242                      42
 1.000000    0.084490      3146.664273                      42
10.000000    0.118200      4089.445753                      42
```


**Final best val by training.lightning.loop_closure_weight:**

```
           count   median      min      max
axis_val                                   
0.000000       1  0.01018  0.01018  0.01018
0.000001       1  0.00971  0.00971  0.00971
0.000010       1  0.00974  0.00974  0.00974
0.000100       1  0.01284  0.01284  0.01284
0.001000       1  0.02416  0.02416  0.02416
0.010000       1  0.03994  0.03994  0.03994
0.100000       1  0.01770  0.01770  0.01770
1.000000       1  0.08449  0.08449  0.08449
10.000000      1  0.11820  0.11820  0.11820
```


**Within-axis-bin std of final_best (median across bins):** nan

**Between-axis-bin std of bin-medians:** 0.03902


#### wmtask_vanilla_old — alpha not logged
