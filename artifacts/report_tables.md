### Dataset composition

| split | images | head inst. | helmet inst. | person inst. | frames with head |
|---|---|---|---|---|---|
| train | 4516 | 4089 | 12750 | 423 | 831 |
| valid | 753 | 785 | 2134 | 50 | 137 |
| test | 1766 | 1803 | 4863 | 142 | 339 |

| class | median box area (% of image) | p10 | p90 | share < 0.1% |
|---|---|---|---|---|
| head | 0.597 | 0.213 | 2.09 | 0.007 |
| helmet | 0.835 | 0.223 | 4.821 | 0.01 |
| person | 8.458 | 1.542 | 33.964 | 0.0 |

Split note: no valid split shipped; every 7th train image (~15%) moved to valid

### Models on the test split

| model | mAP50 | mAP50-95 | AP50 head | AP50 helmet | AP50 person | frame AUC | min cost / 1000 frames |
|---|---|---|---|---|---|---|---|
| finetuned | 0.659 | 0.475 | 0.970 | 0.981 | 0.026 | 0.981 | EUR 1010 |
| coco_zero_shot | 0.009 | 0.006 | 0.000 | 0.000 | 0.026 | 0.676 | EUR 5457 |
| predict_nothing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | EUR 7500 |
| majority_class | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | EUR 7500 |

### Finetuned per-class (box conf >= 0.25, IoU 0.5)

| class | instances | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| head | 1803 | 0.903 | 0.957 | 0.970 | 0.700 |
| helmet | 4863 | 0.918 | 0.963 | 0.981 | 0.710 |
| person | 142 | 0.000 | 0.000 | 0.026 | 0.015 |

Detection confusion matrix (rows = truth, cols = prediction):

| | head | helmet | person | missed |
|---|---|---|---|---|
| **head** | 1722 | 21 | 0 | 60 |
| **helmet** | 13 | 4683 | 0 | 167 |
| **person** | 0 | 0 | 0 | 142 |
| **background** | 176 | 396 | 0 | 0 |

### Operating point and policies (per day = 3000 frames, base rate 5%)

| policy | thresholds | EUR / 1000 | EUR / day | review share | reviewers needed | missed violations/day | false alerts/day |
|---|---|---|---|---|---|---|---|
| do nothing | - | 7500 | 22500 | 0% | 0 | 150 | 0 |
| human checks everything | - | 542 | 1625 | 100% | 2.56 | 0 | 0 |
| model alone | 0.08 / 0.08 | 1010 | 3029 | 0.0% | 0.00 | 6.6 | 127.8 |
| model + review (unconstrained) | 0.00 / 0.86 | 539 | 1617 | 98.5% | 2.53 | 0.0 | 0.0 |
| model + review (<= 1 reviewer) | 0.01 / 0.86 | 567 | 1702 | 14.4% | 0.37 | 3.5 | 0.0 |

### Sensitivity: cost of a missed violation

| cost_fn | best threshold | model alone /1000 | do nothing /1000 | human all /1000 |
|---|---|---|---|---|
| 25 | 0.54 | 567.9 | 1250.0 | 541.7 |
| 50 | 0.31 | 685.2 | 2500.0 | 541.7 |
| 100 | 0.31 | 862.2 | 5000.0 | 541.7 |
| 150 | 0.08 | 1009.8 | 7500.0 | 541.7 |
| 300 | 0.03 | 1277.6 | 15000.0 | 541.7 |
| 600 | 0.03 | 1675.8 | 30000.0 | 541.7 |
| 1000 | 0.03 | 2206.8 | 50000.0 | 541.7 |

### Sensitivity: base rate

| base rate | best threshold | model alone /1000 |
|---|---|---|
| 1% | 0.50 | 328.2 |
| 2% | 0.31 | 516.6 |
| 5% | 0.08 | 1009.8 |
| 10% | 0.03 | 1615.5 |
| 20% | 0.03 | 2689.7 |
