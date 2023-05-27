import torch
import torchmetrics
import torch.nn as nn

import lightning as L
from .load_model import get_model

class LightningSentimentDataset(nn.Module):
    def __init__(self,data,tokenizer,max_len=128):
        self.data=data
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.text=self.data.text
        self.targets=self.data.labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self,index):
        text=str(self.text[index])
        text=" ".join(text.split())

        inputs=self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )

        input_ids=inputs["input_ids"]
        attention_mask=inputs["attention_mask"]

        return {
            "input_ids":torch.tensor(input_ids,dtype=torch.long),
            "attention_mask":torch.tensor(attention_mask,dtype=torch.long),
            "targets":torch.tensor(self.targets[index],dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.text)
    
class LightningModel(L.LightningModule):
    def __init__(self,learning_rate,num_features,num_classes):
      super(LightningModel,self).__init__()
      self.learning_rate=learning_rate
      self.save_hyperparameters(ignore=['model'])
      self.train_acc=torchmetrics.Accuracy(task="multiclass",num_classes=6)
      self.val_acc=torchmetrics.Accuracy(task="multiclass",num_classes=6)
      self.num_features=num_features
      self.num_classes=num_classes
      self.l1,_=get_model()
      
      self.classifier=nn.Sequential(
          nn.Linear(self.num_features,768),
          nn.Tanh(),
          nn.Dropout(0.8),
          nn.Linear(768,self.num_classes)
        )
        
    def forward(self,input_ids,attention_mask):
      output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
      # print(output_1.last_hidden_state.shape)
      hidden_state=output_1[0]
      # assert output_1.last_hidden_state.shape == output_1[0].shape
      pooler=hidden_state[:,0]
      # assert output_1.last_hidden_state.shape == pooler.shape
      logits=self.classifier(pooler)
      return logits 
    
    def training_step(self,batch,true_labels):
      outputs={
          'input_ids':batch["input_ids"],
          'attention_mask':batch["attention_mask"]
      }
      true_labels=batch["targets"]
      logits=self.forward(**outputs)
      # print(logits.shape)
      loss=nn.CrossEntropyLoss()(logits,true_labels)
      self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      with torch.no_grad():
        predicted_labels=torch.argmax(logits,dim=1)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
      return loss
    
    def validation_step(self,batch,true_labels):
      outputs={
          'input_ids':batch["input_ids"],
          'attention_mask':batch["attention_mask"]
      }
      true_labels=batch["targets"]
      logits=self.forward(**outputs)
      loss=nn.CrossEntropyLoss()(logits,true_labels)
      self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      predicted_labels=torch.argmax(logits,dim=1)
      self.val_acc(predicted_labels,true_labels)
      self.log("val_acc",self.val_acc,prog_bar=True)
    
    def configure_optimizers(self):
      optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
      return optimizer    
    