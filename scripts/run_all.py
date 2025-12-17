import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# os.system("python -m train.train_standard")
# os.system("python -m train.train_pgd")
os.system("python -m train.train_pgd_step5")
os.system("python -m eval.eval_autoattack")
