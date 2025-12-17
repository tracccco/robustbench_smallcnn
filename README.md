<<<<<<<< HEAD
# RobustBench-style Evaluation of SmallCNN on CIFAR-10

This repository implements a **RobustBench-style adversarial robustness evaluation pipeline** for a lightweight convolutional neural network (SmallCNN) on CIFAR-10.

The project focuses on **reliable robustness evaluation under strong, standardized attacks** and demonstrates the impact of **PGD adversarial training** when assessed using **AutoAttack**.

---

## Project Structure

```
robustbench_smallcnn/
│
├── models/
│   └── smallcnn.py          # Lightweight CNN model
│
├── attacks/
│   ├── fgsm.py              # FGSM attack implementation
│   ├── pgd.py               # PGD attack implementation
│
├── train/
│   ├── train_standard.py   # Standard (clean) training
│   └── train_pgd.py        # PGD adversarial training
│
├── eval/
│   ├── eval_clean.py       # Clean accuracy evaluation
│   ├── eval_fgsm.py        # FGSM robustness evaluation
│   └── eval_autoattack.py  # ⭐ RobustBench-style AutoAttack evaluation
│
├── scripts/
│   └── run_all.py           # One-command pipeline
│
├── results/
│   ├── smallcnn_clean.pth   # Standard-trained model
│   ├── smallcnn_pgd.pth     # PGD-trained model
│   └── autoattack_*.txt     # Experiment logs
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Python & Dependencies

```bash
conda create -n rb5070 python=3.10
conda activate rb5070
pip install -r requirements.txt
```

> **Note**: All experiments were conducted on **CPU** due to GPU architecture incompatibility with the available PyTorch build.
This does not affect the correctness of robustness evaluation, but increases runtime.

---

## Running Experiments

All experiments can be reproduced with a single command:

### One-command execution

```bash
python scripts/run_all.py
```

This script performs:

1. Train SmallCNN (standard or PGD adversarial training)
2. Evaluate clean accuracy
3. Evaluate robustness using **AutoAttack (L∞, ε = 8/255)**

---

## Experimental Protocol

* **Dataset**: CIFAR-10
* **Model**: SmallCNN (Lightweight custom CNN)
* **Threat Model**: L∞ norm, ε = 8/255
* **Evaluation**: AutoAttack (standard version)

  * APGD-CE
  * APGD-T
  * FAB-T
  * Square Attack

This evaluation protocol follows the **RobustBench standard** for reliable and attack-agnostic robustness measurement.

---

## Results Summary

All reported robust accuracies correspond to the **worst-case accuracy returned by AutoAttack**.

| Training Method | Clean Accuracy | AutoAttack Robust Accuracy |
| --------------- | -------------- | -------------------------- |
| Standard        | ~71%           | ~0.02%                     |
| PGD Training    | ~49%           | ~26%                       |

### Ablation Note

We additionally compared PGD adversarial training with 5 vs 10 iterations.
PGD-10 achieves slightly lower clean accuracy but higher robustness under AutoAttack compared to PGD-5, further confirming that stronger adversarial training improves robustness at the cost of clean performance.


### Key Observations

* Standard training achieves higher clean accuracy but is **almost completely vulnerable** under strong adversarial attacks.
* PGD adversarial training significantly improves robustness at the cost of reduced clean accuracy.
* Later-stage attacks (FAB-T, Square) do not further reduce accuracy after APGD, which is expected since AutoAttack reports the **worst-case result across attacks**.

---

## Discussion

The results illustrate the well-known **robustness–accuracy trade-off** in adversarial machine learning.
While lightweight architectures such as SmallCNN can achieve reasonable clean performance, robustness against adaptive attacks requires explicit adversarial training.

This project confirms that **PGD-based adversarial training** substantially improves robustness even for simple models when evaluated under a standardized benchmark.

---

## Limitations

* AutoAttack evaluation is computationally expensive and was executed on CPU.
* Only a single lightweight architecture was evaluated.

Future work may include GPU-based training, larger architectures,
and alternative defense strategies.

---

## References

* Croce, F., & Hein, M. (2020). *Reliable evaluation of adversarial robustness with AutoAttack*. NeurIPS.
* RobustBench: [https://robustbench.github.io/](https://robustbench.github.io/)

---

## Reproducibility

All experiments are fully reproducible using the provided scripts
and a fixed evaluation protocol.
>>>>>>> 78a34cd (Initial commit: robustbench smallcnn with auto-attack submodule)
