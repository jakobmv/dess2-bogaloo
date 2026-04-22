# SQID Reproduction and DESS Variant Report

## Experimental Setup

All DESS runs use the SQID / ESCI reranking task with the per-query judged candidate list, the corrected gains E=1.0, S=0.1, C=0.01, I=0.0, and NDCG as the headline metric.

Dataset splits used in this repo:

| split | subset | rows | queries |
| --- | --- | --- | --- |
| train | all judged rows | 419,653 | 20,888 |
| train | positive rows used for DESS | 348,537 | 20,888 |
| test | reranking candidate list | 181,701 | 8,956 |

Gain mapping:

| label | gain |
| --- | --- |
| E | 1.0000 |
| S | 0.1000 |
| C | 0.0100 |
| I | 0.0000 |

Frozen feature source: SBERT text embeddings from `sentence-transformers/all-MiniLM-L12-v2` over query text and product titles.

## Part 1: SQID Reproduction

| name | ndcg | num_judgements | num_queries |
| --- | --- | --- | --- |
| random | 0.7465 | 181,701 | 8,956 |
| esci_baseline | 0.8550 | 181,701 | 8,956 |
| sbert_text | 0.8294 | 181,701 | 8,956 |
| clip_text | 0.8106 | 181,701 | 8,956 |
| clip_image | 0.8223 | 181,701 | 8,956 |

Best fusion baseline: `sbert_text_clip_image_score_a0.25` with NDCG `0.8402`.

## Part 2: DESS Variants

| variant | ndcg | delta_vs_sbert | delta_vs_random | delta_vs_esci_baseline | delta_vs_best_fusion | train_rows | train_queries | runtime_seconds | final_loss | final_mu_loss | final_sigma_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_mu_sigma_mlp | 0.8296 | 0.0002 | 0.0831 | -0.0254 | -0.0106 | 348,537 | 20,888 | 43.4810 | 0.0129 | 0.0025 | 0.0237 |
| dual_head_detached_sigma | 0.8078 | -0.0216 | 0.0613 | -0.0472 | -0.0325 | 348,537 | 20,888 | 45.1040 | 0.0101 | 0.0016 | 0.0189 |
| dual_head_query_concat_sigma | 0.8076 | -0.0218 | 0.0611 | -0.0474 | -0.0326 | 348,537 | 20,888 | 44.2360 | 0.0122 | 0.0016 | 0.0231 |
| mlp_joint | 0.8035 | -0.0259 | 0.0571 | -0.0515 | -0.0367 | 348,537 | 20,888 | 44.5240 | 0.0104 | 0.0017 | 0.0193 |

Variant definitions:

| variant | description |
| --- | --- |
| frozen_mu_sigma_mlp | Keeps the frozen query embedding as mu and learns only sigma with a 3-layer MLP. |
| dual_head_detached_sigma | Separate mu and sigma heads; sigma is trained against a detached copy of mu. |
| dual_head_query_concat_sigma | Separate heads; sigma sees the concatenation of the query embedding and detached mu. |
| mlp_joint | One 3-layer MLP jointly predicts the Gaussian mean (mu) and uncertainty (sigma). |

Training configuration:

| variant | batch_size | epochs | learning_rate | weight_decay | beta | alpha | hidden_dim | dropout | device | max_train_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_mu_sigma_mlp | 256 | 3 | 0.0010 | 0.0001 | 1.0000 | 0.5000 | 1024 | 0.1000 | cuda | None |
| dual_head_detached_sigma | 256 | 3 | 0.0010 | 0.0001 | 1.0000 | 0.5000 | 1024 | 0.1000 | cuda | None |
| dual_head_query_concat_sigma | 256 | 3 | 0.0010 | 0.0001 | 1.0000 | 0.5000 | 1024 | 0.1000 | cuda | None |
| mlp_joint | 256 | 3 | 0.0010 | 0.0001 | 1.0000 | 0.5000 | 1024 | 0.1000 | cuda | None |

Artifact locations:

| variant | run_path | checkpoint_path | history_path | metrics_path | metadata_path |
| --- | --- | --- | --- | --- | --- |
| frozen_mu_sigma_mlp | outputs/dess/frozen_mu_sigma_mlp/dess_sbert_text_frozen_mu_sigma_mlp.csv | outputs/dess/frozen_mu_sigma_mlp/dess_sbert_text_frozen_mu_sigma_mlp.pt | outputs/dess/frozen_mu_sigma_mlp/dess_sbert_text_frozen_mu_sigma_mlp.history.csv | outputs/dess/frozen_mu_sigma_mlp/dess_sbert_text_frozen_mu_sigma_mlp.metrics.json | outputs/dess/frozen_mu_sigma_mlp/dess_sbert_text_frozen_mu_sigma_mlp.metadata.json |
| dual_head_detached_sigma | outputs/dess/dual_head_detached_sigma/dess_sbert_text_dual_head_detached_sigma.csv | outputs/dess/dual_head_detached_sigma/dess_sbert_text_dual_head_detached_sigma.pt | outputs/dess/dual_head_detached_sigma/dess_sbert_text_dual_head_detached_sigma.history.csv | outputs/dess/dual_head_detached_sigma/dess_sbert_text_dual_head_detached_sigma.metrics.json | outputs/dess/dual_head_detached_sigma/dess_sbert_text_dual_head_detached_sigma.metadata.json |
| dual_head_query_concat_sigma | outputs/dess/dual_head_query_concat_sigma/dess_sbert_text_dual_head_query_concat_sigma.csv | outputs/dess/dual_head_query_concat_sigma/dess_sbert_text_dual_head_query_concat_sigma.pt | outputs/dess/dual_head_query_concat_sigma/dess_sbert_text_dual_head_query_concat_sigma.history.csv | outputs/dess/dual_head_query_concat_sigma/dess_sbert_text_dual_head_query_concat_sigma.metrics.json | outputs/dess/dual_head_query_concat_sigma/dess_sbert_text_dual_head_query_concat_sigma.metadata.json |
| mlp_joint | outputs/dess/mlp_joint/dess_sbert_text_mlp_joint.csv | outputs/dess/mlp_joint/dess_sbert_text_mlp_joint.pt | outputs/dess/mlp_joint/dess_sbert_text_mlp_joint.history.csv | outputs/dess/mlp_joint/dess_sbert_text_mlp_joint.metrics.json | outputs/dess/mlp_joint/dess_sbert_text_mlp_joint.metadata.json |

Implementation caveats:

| variant | loss_impl | official_probe_ok | official_probe_reason |
| --- | --- | --- | --- |
| frozen_mu_sigma_mlp | dess_updated | False | UnboundLocalError: cannot access local variable 'mu_loss' where it is not associated with a value |
| dual_head_detached_sigma | dess_updated | False | UnboundLocalError: cannot access local variable 'mu_loss' where it is not associated with a value |
| dual_head_query_concat_sigma | dess_updated | False | UnboundLocalError: cannot access local variable 'mu_loss' where it is not associated with a value |
| mlp_joint | dess_updated | False | UnboundLocalError: cannot access local variable 'mu_loss' where it is not associated with a value |

Reference strongest reproduction baseline: `sbert_text_clip_image_score_a0.25` with NDCG `0.8402`.

Initial cache-building pass runtimes captured separately from the warm-cache comparison table:

| variant | runtime_seconds_first_pass |
| --- | --- |
| mlp_joint | 619.0380 |

Best DESS variant in this sweep: `frozen_mu_sigma_mlp` with NDCG `0.8296`.

## Query-Level Deltas

Top query-level changes for the best DESS variant (`frozen_mu_sigma_mlp`) relative to `sbert_text`:

Largest improvements:

| query_id | query | ndcg_baseline | ndcg_variant | delta_ndcg |
| --- | --- | --- | --- | --- |
| 81029 | plastic cup drink sleeve with handle | 0.6107 | 0.9956 | 0.3849 |
| 56863 | kaizen home goods | 0.6120 | 0.9938 | 0.3818 |
| 20874 | butterfly house for outside | 0.6325 | 0.9955 | 0.3630 |
| 5673 | I want a replacement chrome book with a touch screen. So really it would be an upgrade. I want a large screen and a good memory  | 0.6876 | 1.0000 | 0.3124 |
| 60085 | large iscar statie | 0.6617 | 0.9366 | 0.2749 |

Largest regressions:

| query_id | query | ndcg_baseline | ndcg_variant | delta_ndcg |
| --- | --- | --- | --- | --- |
| 54625 | iphone xr usb | 0.9739 | 0.5217 | -0.4522 |
| 267 | - *item not packed in kit - | 1.0000 | 0.6309 | -0.3691 |
| 23450 | cartoon moon rings | 1.0000 | 0.6309 | -0.3691 |
| 28565 | contact lens | 0.9976 | 0.6491 | -0.3484 |
| 49481 | hawaii baby pool | 0.9539 | 0.6221 | -0.3318 |
