import transformers
from transformers import (
    AdamW,
    WarmUp,
    get_linear_schedule_with_warmup,
    DistilBertTokenizer,
    DistilBertModel
)

import streamlit as st

from .Config import Config

@st.cache_data()
def get_model():
  tokenizer=DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
  SPECIAL_TOKENS_DICT={
     'sep_token':'[SEP]', 
     'pad_token':'[PAD]',
     'cls_token':'[CLS]' 
  }
  tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
  model=DistilBertModel.from_pretrained(Config.MODEL_NAME)
  model.resize_token_embeddings(len(tokenizer))
  return model,tokenizer