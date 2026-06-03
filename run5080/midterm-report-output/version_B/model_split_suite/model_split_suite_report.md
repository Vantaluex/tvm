# Model Split Suite Report

- Version: B
- Tuning policy: reuse
- Suite mode: both
- Frequency tags: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

## Suite overview

| experiment | experiment_kind | allowed_freq_tags | n_models | n_rows | avg_train_mape_pct | avg_val_mape_pct | avg_test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convnextv2_tiny_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 100000 | 3.1377% | 7.1461% | 7.3504% |
| deberta_v3_base_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99995 | 9.7116% | 15.8575% | 12.7750% |
| deepseekr1_qwen_14b_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99742 | 0.8875% | 15.4358% | 40.0549% |
| densenet169_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 100000 | 1.1104% | 7.5089% | 7.5898% |
| exaone3.5_7.8b_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99994 | 1.5761% | 17.1289% | 18.8083% |
| exaone_deep_7.8B_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 95925 | 4.0869% | 7.1908% | 30.1376% |
| fully_pooled_model_split | fully_pooled_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 12 | 1194984 | 5.5322% | 15.4643% | 13.0144% |
| llama_3.1_8b_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99729 | 2.4170% | 18.5096% | 57.0719% |
| mask2former_swin_small_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99995 | 2.4991% | 8.4969% | 5.9266% |
| mobilenetv3large_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 100000 | 1.4408% | 7.2772% | 10.3382% |
| modernbert_base_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 100000 | 0.5742% | 16.9588% | 8.6986% |
| qwen2.5_3b_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99951 | 6.5363% | 11.2335% | 9.7784% |
| qwen2.5_9B_model_split | single_model_5freq | 1500x0,2295x0,2595x0,2617x0,975x0 | 1 | 99653 | 3.2455% | 15.8115% | 15.8887% |

## convnextv2_tiny_model_split

- Kind: single_model_5freq
- Models: convnextv2-tiny
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | convnextv2-tiny | convnextv2-tiny | 82925 | 9030 | 8045 | 64 | 7.1461% | 3.1377% | 7.1461% | 7.3504% |

## deberta_v3_base_model_split

- Kind: single_model_5freq
- Models: deberta-v3-base
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | deberta-v3-base | deberta-v3-base | 41845 | 26495 | 31655 | 12 | 15.8575% | 9.7116% | 15.8575% | 12.7750% |

## deepseekr1_qwen_14b_model_split

- Kind: single_model_5freq
- Models: deepseekr1-qwen-14b
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | deepseekr1-qwen-14b | deepseekr1-qwen-14b | 96227 | 3495 | 20 | 416 | 15.4358% | 0.8875% | 15.4358% | 40.0549% |

## densenet169_model_split

- Kind: single_model_5freq
- Models: densenet169
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | densenet169 | densenet169 | 80865 | 12525 | 6610 | 470 | 7.5089% | 1.1104% | 7.5089% | 7.5898% |

## exaone_deep_7.8B_model_split

- Kind: single_model_5freq
- Models: exaone-deep-7.8B
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | exaone-deep-7.8B | exaone-deep-7.8B | 95560 | 40 | 325 | 137 | 7.1908% | 4.0869% | 7.1908% | 30.1376% |

## exaone3.5_7.8b_model_split

- Kind: single_model_5freq
- Models: exaone3.5-7.8b
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | exaone3.5-7.8b | exaone3.5-7.8b | 49575 | 38909 | 11510 | 54 | 17.1289% | 1.5761% | 17.1289% | 18.8083% |

## llama_3.1_8b_model_split

- Kind: single_model_5freq
- Models: llama-3.1-8b
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | llama-3.1-8b | llama-3.1-8b | 50772 | 48632 | 325 | 95 | 18.5096% | 2.4170% | 18.5096% | 57.0719% |

## mask2former_swin_small_model_split

- Kind: single_model_5freq
- Models: mask2former-swin-small
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | mask2former-swin-small | mask2former-swin-small | 72310 | 16060 | 11625 | 132 | 8.4969% | 2.4991% | 8.4969% | 5.9266% |

## mobilenetv3large_model_split

- Kind: single_model_5freq
- Models: mobilenetv3large
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | mobilenetv3large | mobilenetv3large | 85575 | 7065 | 7360 | 53 | 7.2772% | 1.4408% | 7.2772% | 10.3382% |

## modernbert_base_model_split

- Kind: single_model_5freq
- Models: modernbert-base
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | modernbert-base | modernbert-base | 84640 | 10 | 15350 | 208 | 16.9588% | 0.5742% | 16.9588% | 8.6986% |

## qwen2.5_3b_model_split

- Kind: single_model_5freq
- Models: qwen2.5-3b
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | qwen2.5-3b | qwen2.5-3b | 91951 | 2250 | 5750 | 47 | 11.2335% | 6.5363% | 11.2335% | 9.7784% |

## qwen2.5_9B_model_split

- Kind: single_model_5freq
- Models: qwen2.5-9B
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | qwen2.5-9B | qwen2.5-9B | 64060 | 26956 | 8637 | 124 | 15.8115% | 3.2455% | 15.8115% | 15.8887% |

## fully_pooled_model_split

- Kind: fully_pooled_5freq
- Models: convnextv2-tiny, deberta-v3-base, deepseekr1-qwen-14b, densenet169, exaone-deep-7.8B, exaone3.5-7.8b, llama-3.1-8b, mask2former-swin-small, mobilenetv3large, modernbert-base, qwen2.5-3b, qwen2.5-9B
- Frequencies: 2295x0, 2617x0, 2595x0, 975x0, 1500x0

| domain | train_models | test_models | train_rows | val_rows | test_rows | best_iteration | best_val_mape_pct | train_mape_pct | val_mape_pct | test_mape_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | deberta-v3-base,densenet169,exaone-deep-7.8B,llama-3.1-8b,mobilenetv3large,qwen2.5-3b,qwen2.5-9B | convnextv2-tiny,deepseekr1-qwen-14b,exaone3.5-7.8b,modernbert-base | 601581 | 93672 | 399736 | 144 | 15.4643% | 5.5322% | 15.4643% | 13.0144% |
