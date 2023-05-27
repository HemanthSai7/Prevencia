import warnings

import torch

from .Config import Config
from .load_model import get_model
from .dataloader import LightningModel

SEED=42
warnings.filterwarnings(action="ignore",category=UserWarning)



lightning_model=LightningModel(learning_rate=Config.lr,num_features=768,num_classes=6)    
labeldict={"none":0,"homophobia":1,"Bullying":2,"Hate_Speech":3,"Racism":4,"sexism":5}

_,tokenizer=get_model()
def predict(text):
    encoded_text=tokenizer.encode_plus(
        text,
        max_length=100,
        add_special_tokens=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids=encoded_text['input_ids']
    attention_mask=encoded_text['attention_mask']
    model=lightning_model.load_from_checkpoint(checkpoint_path="src/SentimentAnalysis/models/distilBERT_v1.ckpt").to('cpu')
    model.eval()
    with torch.no_grad():
      output=model(input_ids,attention_mask).to('cpu')
      probabilities=torch.softmax(output,dim=1)
      confidence, predicted_class=torch.max(probabilities,dim=1)
      key = list(filter(lambda x: labeldict[x] == predicted_class, labeldict))[0]
      
      return predicted_class,confidence,key
