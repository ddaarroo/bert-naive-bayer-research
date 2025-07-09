import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class BertConversationalLanguageClassificationModel(nn.Module): 
    def __init__(self, model):
        super().__init__()
        self.embedding = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)
    
    def forward(self, input_ids, attention_mask): 
        output = self.embedding(input_ids, attention_mask)
        embeddings = output.logits

        a = F.log_softmax(embeddings, dim=1)

        return a