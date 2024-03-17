import os
# 加载数据集
from datasets import load_dataset, DatasetDict
# Robert分词
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"




dataset_train = load_dataset('csv', data_files='data/train.csv', encoding='utf-8')
dataset_test = load_dataset('csv', data_files='data/test.csv', encoding='utf-8')
dataset_valid = load_dataset('csv', data_files='data/valid.csv', encoding='utf-8')
dataset = DatasetDict({'train': dataset_train, 'test': dataset_test, 'validation': dataset_valid})


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def encode_batch(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=72 , return_tensors="pt")

dataset = {split: dataset[split].map(encode_batch, batched=True) for split in dataset.keys()}
dataset['test']['train'] = dataset['test']['train'].remove_columns('text')
dataset['train']['train'] = dataset['train']['train'].remove_columns('text')
dataset['validation']['train'] = dataset['validation']['train'].remove_columns('text')


# 定义模型
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

optimizer = AdamW(model.parameters(), lr=1e-5)


train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=16)
valid_dataloader = DataLoader(dataset["validation"], batch_size=16)



epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")


