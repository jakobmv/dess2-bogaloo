# Single-Target DESS Report

## Dataset

Dataset: UCI Gas Turbine CO and NOx Emission Data Set.

Citation: Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95.

Inputs: 9 tabular features.

Targets: CO, NOX.

Split protocol: 2011-2013 used as the training/cross-validation pool, 2014-2015 used as test; the tail of the train pool is held out as validation while preserving chronology.

| split | rows |
| --- | --- |
| train | 17,753 |
| validation | 4,438 |
| test | 14,542 |

## Aggregate Results

| variant | rmse | mae | r2 | mean_nll | rmse_co | rmse_nox | r2_co | r2_nox | runtime_s | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_mu_sigma_mlp | 6.5205 ± 0.0000 | 4.0682 ± 0.0000 | 0.2119 ± 0.0000 | 11.8165 ± 1.2993 | 2.0216 | 8.9970 | 0.1474 | 0.2764 | 1.5640 ± 0.0219 | 3 |
| mlp_joint | 8.8084 ± 0.2521 | 6.0308 ± 0.1777 | 0.1101 ± 0.0375 | 8.7429 ± 0.2239 | 1.4009 | 12.3778 | 0.5905 | -0.3704 | 2.8390 ± 1.1400 | 3 |
| dual_head_detached_sigma | 8.8842 ± 0.2131 | 6.0952 ± 0.1613 | 0.1000 ± 0.0345 | 9.4957 ± 0.5698 | 1.3945 | 12.4865 | 0.5943 | -0.3943 | 3.0990 ± 0.7314 | 3 |
| dual_head_query_concat_sigma | 8.8842 ± 0.2131 | 6.0952 ± 0.1613 | 0.1000 ± 0.0345 | 8.8444 ± 0.7327 | 1.3945 | 12.4865 | 0.5943 | -0.3943 | 3.1323 ± 0.7185 | 3 |

Best variant by mean test RMSE: `frozen_mu_sigma_mlp`.

## Per-Run Results

| variant | seed | test_rmse | test_mae | test_r2 | test_mean_nll | best_epoch | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dual_head_detached_sigma | 42 | 8.6398 | 5.9090 | 0.1391 | 9.4581 | 8 | 2.8680 |
| dual_head_detached_sigma | 43 | 9.0311 | 6.1876 | 0.0739 | 10.0834 | 6 | 2.5110 |
| dual_head_detached_sigma | 44 | 8.9817 | 6.1890 | 0.0870 | 8.9457 | 15 | 3.9180 |
| dual_head_query_concat_sigma | 42 | 8.6398 | 5.9090 | 0.1391 | 8.8112 | 8 | 2.9010 |
| dual_head_query_concat_sigma | 43 | 9.0311 | 6.1876 | 0.0739 | 9.5932 | 6 | 2.5580 |
| dual_head_query_concat_sigma | 44 | 8.9817 | 6.1890 | 0.0870 | 8.1289 | 15 | 3.9380 |
| frozen_mu_sigma_mlp | 42 | 6.5205 | 4.0682 | 0.2119 | 11.6426 | 1 | 1.5590 |
| frozen_mu_sigma_mlp | 43 | 6.5205 | 4.0682 | 0.2119 | 10.6130 | 1 | 1.5880 |
| frozen_mu_sigma_mlp | 44 | 6.5205 | 4.0682 | 0.2119 | 13.1940 | 1 | 1.5450 |
| mlp_joint | 42 | 8.5682 | 5.8797 | 0.1431 | 8.5889 | 8 | 4.1380 |
| mlp_joint | 43 | 9.0709 | 6.2266 | 0.0693 | 8.9998 | 6 | 2.3740 |
| mlp_joint | 44 | 8.7861 | 5.9860 | 0.1179 | 8.6401 | 4 | 2.0050 |

## Artifacts

| variant | seed | predictions_path | metrics_path | history_path | checkpoint_path |
| --- | --- | --- | --- | --- | --- |
| dual_head_detached_sigma | 42 | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_42/gas_turbine_dual_head_detached_sigma_seed42.predictions.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_42/gas_turbine_dual_head_detached_sigma_seed42.metrics.json | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_42/gas_turbine_dual_head_detached_sigma_seed42.history.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_42/gas_turbine_dual_head_detached_sigma_seed42.pt |
| dual_head_detached_sigma | 43 | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_43/gas_turbine_dual_head_detached_sigma_seed43.predictions.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_43/gas_turbine_dual_head_detached_sigma_seed43.metrics.json | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_43/gas_turbine_dual_head_detached_sigma_seed43.history.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_43/gas_turbine_dual_head_detached_sigma_seed43.pt |
| dual_head_detached_sigma | 44 | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_44/gas_turbine_dual_head_detached_sigma_seed44.predictions.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_44/gas_turbine_dual_head_detached_sigma_seed44.metrics.json | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_44/gas_turbine_dual_head_detached_sigma_seed44.history.csv | outputs/single_target_gas_turbine/dual_head_detached_sigma/seed_44/gas_turbine_dual_head_detached_sigma_seed44.pt |
| dual_head_query_concat_sigma | 42 | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_42/gas_turbine_dual_head_query_concat_sigma_seed42.predictions.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_42/gas_turbine_dual_head_query_concat_sigma_seed42.metrics.json | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_42/gas_turbine_dual_head_query_concat_sigma_seed42.history.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_42/gas_turbine_dual_head_query_concat_sigma_seed42.pt |
| dual_head_query_concat_sigma | 43 | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_43/gas_turbine_dual_head_query_concat_sigma_seed43.predictions.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_43/gas_turbine_dual_head_query_concat_sigma_seed43.metrics.json | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_43/gas_turbine_dual_head_query_concat_sigma_seed43.history.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_43/gas_turbine_dual_head_query_concat_sigma_seed43.pt |
| dual_head_query_concat_sigma | 44 | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_44/gas_turbine_dual_head_query_concat_sigma_seed44.predictions.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_44/gas_turbine_dual_head_query_concat_sigma_seed44.metrics.json | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_44/gas_turbine_dual_head_query_concat_sigma_seed44.history.csv | outputs/single_target_gas_turbine/dual_head_query_concat_sigma/seed_44/gas_turbine_dual_head_query_concat_sigma_seed44.pt |
| frozen_mu_sigma_mlp | 42 | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_42/gas_turbine_frozen_mu_sigma_mlp_seed42.predictions.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_42/gas_turbine_frozen_mu_sigma_mlp_seed42.metrics.json | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_42/gas_turbine_frozen_mu_sigma_mlp_seed42.history.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_42/gas_turbine_frozen_mu_sigma_mlp_seed42.pt |
| frozen_mu_sigma_mlp | 43 | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_43/gas_turbine_frozen_mu_sigma_mlp_seed43.predictions.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_43/gas_turbine_frozen_mu_sigma_mlp_seed43.metrics.json | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_43/gas_turbine_frozen_mu_sigma_mlp_seed43.history.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_43/gas_turbine_frozen_mu_sigma_mlp_seed43.pt |
| frozen_mu_sigma_mlp | 44 | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_44/gas_turbine_frozen_mu_sigma_mlp_seed44.predictions.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_44/gas_turbine_frozen_mu_sigma_mlp_seed44.metrics.json | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_44/gas_turbine_frozen_mu_sigma_mlp_seed44.history.csv | outputs/single_target_gas_turbine/frozen_mu_sigma_mlp/seed_44/gas_turbine_frozen_mu_sigma_mlp_seed44.pt |
| mlp_joint | 42 | outputs/single_target_gas_turbine/mlp_joint/seed_42/gas_turbine_mlp_joint_seed42.predictions.csv | outputs/single_target_gas_turbine/mlp_joint/seed_42/gas_turbine_mlp_joint_seed42.metrics.json | outputs/single_target_gas_turbine/mlp_joint/seed_42/gas_turbine_mlp_joint_seed42.history.csv | outputs/single_target_gas_turbine/mlp_joint/seed_42/gas_turbine_mlp_joint_seed42.pt |
| mlp_joint | 43 | outputs/single_target_gas_turbine/mlp_joint/seed_43/gas_turbine_mlp_joint_seed43.predictions.csv | outputs/single_target_gas_turbine/mlp_joint/seed_43/gas_turbine_mlp_joint_seed43.metrics.json | outputs/single_target_gas_turbine/mlp_joint/seed_43/gas_turbine_mlp_joint_seed43.history.csv | outputs/single_target_gas_turbine/mlp_joint/seed_43/gas_turbine_mlp_joint_seed43.pt |
| mlp_joint | 44 | outputs/single_target_gas_turbine/mlp_joint/seed_44/gas_turbine_mlp_joint_seed44.predictions.csv | outputs/single_target_gas_turbine/mlp_joint/seed_44/gas_turbine_mlp_joint_seed44.metrics.json | outputs/single_target_gas_turbine/mlp_joint/seed_44/gas_turbine_mlp_joint_seed44.history.csv | outputs/single_target_gas_turbine/mlp_joint/seed_44/gas_turbine_mlp_joint_seed44.pt |
