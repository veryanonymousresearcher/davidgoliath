

<!-- T1_DISTILL_BY_DATASET_START -->

| log       | metric   |    diff |    err |
|:----------|:---------|--------:|-------:|
| BPI12     | Accuracy | -0.04   | 0.0465 |
| BPI12     | F1       | -0.0672 | 0.0738 |
| BPI15     | Accuracy | -0.0497 | 0.0412 |
| BPI15     | F1       | -0.0275 | 0.0213 |
| BPI17     | Accuracy | -0.0195 | 0.0302 |
| BPI17     | F1       | -0.0495 | 0.0718 |
| BPI19     | Accuracy | -0.0008 | 0.0033 |
| BPI19     | F1       | -0.0014 | 0.0119 |
| BPI20_PTC | Accuracy | -0.0743 | 0.0718 |
| BPI20_PTC | F1       | -0.0877 | 0.0596 |
| BPI20_RfP | Accuracy | -0.014  | 0.0256 |
| BPI20_RfP | F1       | -0.0445 | 0.0242 |
| BPI20_TPD | Accuracy | -0.0202 | 0.0209 |
| BPI20_TPD | F1       | -0.0463 | 0.0433 |

<!-- T1_DISTILL_BY_DATASET_END -->


<!-- T2_DISTILL_BY_SIZE_START -->

|      size | metric   |    diff |    err |
|----------:|:---------|--------:|-------:|
|  128_12_2 | Accuracy | -0.0224 | 0.4061 |
|  128_12_2 | F1       | -0.0287 | 0.3351 |
|   128_4_2 | Accuracy | -0.0168 | 0.4114 |
|   128_4_2 | F1       | -0.0168 | 0.34   |
|  256_12_4 | Accuracy | -0.0489 | 0.4051 |
|  256_12_4 | F1       | -0.0643 | 0.3211 |
|   256_4_4 | Accuracy | -0.0241 | 0.4125 |
|   256_4_4 | F1       | -0.0383 | 0.3281 |
|  512_12_8 | Accuracy | -0.0683 | 0.4084 |
|  512_12_8 | F1       | -0.0938 | 0.3042 |
|   512_4_8 | Accuracy | -0.026  | 0.4111 |
|   512_4_8 | F1       | -0.0393 | 0.3221 |
|    64_2_1 | Accuracy |  0.0015 | 0.3672 |
|    64_2_1 | F1       | -0.0079 | 0.3191 |
|    64_4_1 | Accuracy | -0.0023 | 0.376  |
|    64_4_1 | F1       | -0.0133 | 0.3228 |
| 768_12_12 | Accuracy | -0.0585 | 0.4077 |
| 768_12_12 | F1       | -0.096  | 0.2962 |
|  768_4_12 | Accuracy | -0.0462 | 0.4129 |
|  768_4_12 | F1       | -0.0646 | 0.3184 |

<!-- T2_DISTILL_BY_SIZE_END -->


<!-- T3_LLM_VS_REFS_START -->

| log       | metric   | ref       |    diff |    err |
|:----------|:---------|:----------|--------:|-------:|
| BPI12     | Accuracy | vs Argmax |  0.2191 | 0.045  |
| BPI12     | F1       | vs Argmax |  0.3611 | 0.0202 |
| BPI15     | Accuracy | vs Argmax |  0.027  | 0.0582 |
| BPI15     | F1       | vs Argmax | -0.02   | 0.0296 |
| BPI17     | Accuracy | vs Argmax |  0.1367 | 0.0705 |
| BPI17     | F1       | vs Argmax |  0.2035 | 0.0243 |
| BPI19     | Accuracy | vs Argmax | -0.0225 | 0.0671 |
| BPI19     | F1       | vs Argmax |  0.0721 | 0.03   |
| BPI20_PTC | Accuracy | vs Argmax |  0.01   | 0.0355 |
| BPI20_PTC | F1       | vs Argmax | -0.0341 | 0.0682 |
| BPI20_RfP | Accuracy | vs Argmax | -0.0083 | 0.0151 |
| BPI20_RfP | F1       | vs Argmax | -0.02   | 0.0333 |
| BPI20_TPD | Accuracy | vs Argmax | -0.0669 | 0.0382 |
| BPI20_TPD | F1       | vs Argmax | -0.0522 | 0.0473 |
| BPI12     | Accuracy | vs LSTM   |  0.1384 | 0.0159 |
| BPI12     | F1       | vs LSTM   |  0.072  | 0.0235 |
| BPI15     | Accuracy | vs LSTM   | -0.0276 | 0.0602 |
| BPI15     | F1       | vs LSTM   | -0.0258 | 0.0307 |
| BPI17     | Accuracy | vs LSTM   |  0.054  | 0.0124 |
| BPI17     | F1       | vs LSTM   |  0.0163 | 0.0261 |
| BPI19     | Accuracy | vs LSTM   |  0.1682 | 0.0285 |
| BPI19     | F1       | vs LSTM   | -0.0623 | 0.0307 |
| BPI20_PTC | Accuracy | vs LSTM   |  0.0773 | 0.0353 |
| BPI20_PTC | F1       | vs LSTM   | -0.0568 | 0.0689 |
| BPI20_RfP | Accuracy | vs LSTM   |  0.0336 | 0.0173 |
| BPI20_RfP | F1       | vs LSTM   | -0.0215 | 0.0385 |
| BPI20_TPD | Accuracy | vs LSTM   |  0.1029 | 0.0357 |
| BPI20_TPD | F1       | vs LSTM   | -0.0971 | 0.0478 |

<!-- T3_LLM_VS_REFS_END -->
