# Q1 / Q2: per-run plateau and post-best walltime


## lorenz_current (n=21)

```
        n_epochs  runtime_total_s  per_epoch_s_median  best_traj_val  best_traj_val_epoch  best_traj_val_runtime_s  walltime_past_best_s  walltime_past_best_frac  first_traj_val_within_0.05_epoch  sat@0.05_train/recon_loss  sat@0.05_train/latent_pred_loss  sat@0.05_train/loop_closure_loss  sat@0.05_train/trajectory_loss
median   109.000        10602.172              93.777          0.006              102.000                10070.762               364.525                    0.034                            98.000                     52.000                           27.000                            28.000                          25.000
mean     105.905        10294.738              96.194          0.007               98.762                 9773.063               521.675                    0.050                            93.762                     53.857                           44.048                            31.143                          22.952
min       58.000         5309.065              89.908          0.005               55.000                 5217.324                 0.000                    0.000                            52.000                     38.000                           19.000                             0.000                           9.000
max      115.000        10657.658             110.326          0.010              114.000                10632.770              1784.283                    0.168                           106.000                     78.000                          100.000                           110.000                          49.000
```


**Median walltime past best traj val:** 3.4% (n=21)

**Median epoch at which traj val is within 5% of best:** 98 of 109 epochs total (89.9% of run length)


## lorenz_current_obsnoise001 (n=21)

```
        n_epochs  runtime_total_s  per_epoch_s_median  best_traj_val  best_traj_val_epoch  best_traj_val_runtime_s  walltime_past_best_s  walltime_past_best_frac  first_traj_val_within_0.05_epoch  sat@0.05_train/recon_loss  sat@0.05_train/latent_pred_loss  sat@0.05_train/loop_closure_loss  sat@0.05_train/trajectory_loss
median   105.000        10598.996             101.162          0.001               101.00                10412.795               203.153                    0.019                           101.000                     88.000                           56.000                             1.000                          34.000
mean     107.476        10599.552              98.775          0.001               102.81                10340.773               258.778                    0.024                           102.524                     74.952                           58.143                            19.476                          43.048
min      103.000        10556.286              89.023          0.001               100.00                 9092.360                 0.000                    0.000                           100.000                     36.000                           30.000                             0.000                          19.000
max      119.000        10643.238             102.963          0.003               116.00                10605.591              1518.513                    0.143                           116.000                    105.000                           88.000                            96.000                         102.000
```


**Median walltime past best traj val:** 1.9% (n=21)

**Median epoch at which traj val is within 5% of best:** 101 of 105 epochs total (96.2% of run length)


## wmtask_vanilla_old (n=9)

```
        n_epochs  runtime_total_s  per_epoch_s_median  best_traj_val  best_traj_val_epoch  best_traj_val_runtime_s  walltime_past_best_s  walltime_past_best_frac  first_traj_val_within_0.05_epoch  sat@0.05_train/recon_loss  sat@0.05_train/latent_pred_loss  sat@0.05_train/loop_closure_loss  sat@0.05_train/trajectory_loss
median    12.000         2596.174             215.392          0.018                9.000                 2382.287               213.886                    0.082                             9.000                        NaN                              NaN                               NaN                             NaN
mean      12.556         3009.900             237.792          0.036                9.556                 2774.336               235.564                    0.086                             9.333                        NaN                              NaN                               NaN                             NaN
min        8.000         1711.417             211.708          0.010                5.000                 1497.835               209.690                    0.042                             4.000                        NaN                              NaN                               NaN                             NaN
max       23.000         5012.708             314.836          0.118               20.000                 4800.556               310.926                    0.125                            20.000                        NaN                              NaN                               NaN                             NaN
```


**Median walltime past best traj val:** 8.2% (n=9)

**Median epoch at which traj val is within 5% of best:** 9 of 12 epochs total (75.0% of run length)
