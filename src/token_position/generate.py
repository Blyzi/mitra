from pathlib import Path
import random
import sys
from typing import Literal
import torch
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, Path.cwd().as_posix())

from src.utils.functions import get_top_k
from src.utils.icl import ICLDataset
from src.utils.evaluation import eval_bleu, eval_chrf
from src.utils.get_model import get_model
from src.utils.fasttext import get_language, flores_langs
from src.utils.add_vectors import add_vectors