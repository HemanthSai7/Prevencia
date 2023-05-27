import os
import random

import numpy as np

import torch

class Config:
  batch_size=32
  num_workers=4
  lr=0.00003
  epochs=200
  load_weights_path="models/"
  save_file_name="model_weights_distilbert_lightning_v1"
  MODEL_NAME="distilbert-base-uncased"
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"]=str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.benchmark=True
