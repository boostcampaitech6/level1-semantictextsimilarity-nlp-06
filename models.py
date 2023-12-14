from torch import nn 
from transformers import AutoModelForSequenceClassification, AutoModel

class Model(nn.Module):
    def __init__(self, model_name='jhgan/ko-sroberta-sts', input_dim=768, num_labels=1):
        super(Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.sigmoid = nn.Sigmoid()
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)
    
    def forward(self, input_ids, type_ids, mask):
        pooler = self.model(input_ids=input_ids, token_type_ids=type_ids, attention_mask=mask)
        logits = pooler['logits']
        return self.sigmoid(logits)
