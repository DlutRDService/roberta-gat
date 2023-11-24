import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 如果是自动模型类，替换为相应的模型类
model = RobertaForSequenceClassification.from_pretrained('./results/checkpoint-100')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encoded_inputs = tokenizer('Despite the benefits of counterfactual explanations, the user study showed that '
                           'feature attribution explanations are still more widely accepted than counterfactuals, '
                           'possibly due to the greater familiarity with the former and the novelty of the '
                           'latter', return_tensors='pt')

predictions = model(**encoded_inputs)  #
print(predictions)
logits = predictions.logits
print(logits)
predicted_label_indices = torch.argmax(logits, dim=-1)
print('label: ' + str(predicted_label_indices.item()))
