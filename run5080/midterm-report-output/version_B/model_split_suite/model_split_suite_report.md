# Model Split Suite Report

- Version: B
- Tuning policy: reuse
- Suite mode: both
- Frequency tags: 2295x0, 2617x0, 2595x0

## Suite overview

| experiment | experiment_kind | allowed_freq_tags | n_models | n_rows | avg_train_mape_pct | avg_val_mape_pct | avg_test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convnextv2_tiny_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 60000 | 3.7496% | 9.6299% | 8.7532% |
| deberta_v3_base_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59997 | 3.6890% | 17.1705% | 7.5173% |
| deepseekr1_qwen_14b_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59918 | 0.6964% | 13.0174% | 41.8968% |
| densenet169_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 60000 | 2.5174% | 8.7326% | 8.4912% |
| exaone3.5_7.8b_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59997 | 2.0479% | 18.5782% | 13.2401% |
| exaone_deep_7.8B_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59873 | 4.5048% | 4.5179% | 46.3888% |
| fully_pooled_model_split | fully_pooled_3freq | 2295x0,2595x0,2617x0 | 13 | 779536 | 3.5679% | 12.8838% | 15.1203% |
| llama_3.1_8b_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59919 | 1.0509% | 20.7041% | 45.5829% |
| mask2former_swin_small_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59997 | 3.1317% | 9.0454% | 7.3183% |
| mobilenetv3large_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 60000 | 3.5503% | 8.1940% | 11.2011% |
| modernbert_base_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 60000 | 0.7659% | 18.5207% | 10.5194% |
| qwen2.5_3b_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59981 | 8.0421% | 12.5990% | 10.2188% |
| qwen2.5_9B_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59861 | 1.7023% | 49.8329% | 18.5296% |
| resnet50_model_split | single_model_3freq | 2295x0,2595x0,2617x0 | 1 | 59993 | 10.2506% | 10.8964% | 13.0858% |

## convnextv2_tiny_model_split

- Kind: single_model_3freq
- Models: convnextv2-tiny
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | convnextv2-tiny | convnextv2-tiny | 49962 | 5211 | 4827 | 55 | 9.6299% | 3.7496% | 9.6299% | 8.7532% |

## deberta_v3_base_model_split

- Kind: single_model_3freq
- Models: deberta-v3-base
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | deberta-v3-base | deberta-v3-base | 25107 | 15897 | 18993 | 21 | 17.1705% | 3.6890% | 17.1705% | 7.5173% |

## deepseekr1_qwen_14b_model_split

- Kind: single_model_3freq
- Models: deepseekr1-qwen-14b
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | deepseekr1-qwen-14b | deepseekr1-qwen-14b | 57809 | 2097 | 12 | 236 | 13.0174% | 0.6964% | 13.0174% | 41.8968% |

## densenet169_model_split

- Kind: single_model_3freq
- Models: densenet169
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | densenet169 | densenet169 | 49059 | 7515 | 3426 | 123 | 8.7326% | 2.5174% | 8.7326% | 8.4912% |

## exaone_deep_7.8B_model_split

- Kind: single_model_3freq
- Models: exaone-deep-7.8B
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | exaone-deep-7.8B | exaone-deep-7.8B | 59654 | 24 | 195 | 64 | 4.5179% | 4.5048% | 4.5179% | 46.3888% |

## exaone3.5_7.8b_model_split

- Kind: single_model_3freq
- Models: exaone3.5-7.8b
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | exaone3.5-7.8b | exaone3.5-7.8b | 29745 | 23346 | 6906 | 34 | 18.5782% | 2.0479% | 18.5782% | 13.2401% |

## llama_3.1_8b_model_split

- Kind: single_model_3freq
- Models: llama-3.1-8b
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | llama-3.1-8b | llama-3.1-8b | 30481 | 29243 | 195 | 744 | 20.7041% | 1.0509% | 20.7041% | 45.5829% |

## mask2former_swin_small_model_split

- Kind: single_model_3freq
- Models: mask2former-swin-small
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | mask2former-swin-small | mask2former-swin-small | 46473 | 6567 | 6957 | 29 | 9.0454% | 3.1317% | 9.0454% | 7.3183% |

## mobilenetv3large_model_split

- Kind: single_model_3freq
- Models: mobilenetv3large
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | mobilenetv3large | mobilenetv3large | 51345 | 4239 | 4416 | 33 | 8.1940% | 3.5503% | 8.1940% | 11.2011% |

## modernbert_base_model_split

- Kind: single_model_3freq
- Models: modernbert-base
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | modernbert-base | modernbert-base | 50784 | 6 | 9210 | 177 | 18.5207% | 0.7659% | 18.5207% | 10.5194% |

## qwen2.5_3b_model_split

- Kind: single_model_3freq
- Models: qwen2.5-3b
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | qwen2.5-3b | qwen2.5-3b | 55181 | 1350 | 3450 | 29 | 12.5990% | 8.0421% | 12.5990% | 10.2188% |

## qwen2.5_9B_model_split

- Kind: single_model_3freq
- Models: qwen2.5-9B
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | qwen2.5-9B | qwen2.5-9B | 38488 | 198 | 21175 | 106 | 49.8329% | 1.7023% | 49.8329% | 18.5296% |

## resnet50_model_split

- Kind: single_model_3freq
- Models: resnet50
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | resnet50 | resnet50 | 48665 | 6720 | 4608 | 17 | 10.8964% | 10.2506% | 10.8964% | 13.0858% |

## fully_pooled_model_split

- Kind: fully_pooled_3freq
- Models: convnextv2-tiny, deberta-v3-base, deepseekr1-qwen-14b, densenet169, exaone-deep-7.8B, exaone3.5-7.8b, llama-3.1-8b, mask2former-swin-small, mobilenetv3large, modernbert-base, qwen2.5-3b, qwen2.5-9B, resnet50
- Frequencies: 2295x0, 2617x0, 2595x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | convnextv2-tiny,deberta-v3-base,densenet169,exaone-deep-7.8B,llama-3.1-8b,qwen2.5-3b,qwen2.5-9B,resnet50 | deepseekr1-qwen-14b,exaone3.5-7.8b,mobilenetv3large,modernbert-base | 447402 | 32222 | 239915 | 80 | 12.8838% | 3.5679% | 12.8838% | 15.1203% |
