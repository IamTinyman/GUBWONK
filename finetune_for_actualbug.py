import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 加载数据 选择列 结合列 
df = pd.read_csv('bug_report.csv') 
df = df[['merged_text', 'comments', 'nonbug/bug_nonbug', 'nonbug/bug_bug', 'nonbug/bug_invalid']]
df['text'] = df['merged_text'] + " " + df['comments']

# 标签列
def create_label(row):
    if row['nonbug/bug_nonbug']:
        return 0
    elif row['nonbug/bug_bug']:
        return 1
    elif row['nonbug/bug_invalid']:
        return 2
    else:
        return -1  # 处理无效标签
df['label'] = df.apply(create_label, axis=1)
df = df[df['label'] != -1] 

#分割 80% - 20%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

#  转换为 Hugging Face 的 Dataset 格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
# 创建 DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 分词器 数据预处理
tokenizer = RobertaTokenizer.from_pretrained('./roberta_bug_report_model')  
def preprocess_function(examples):
    texts = [str(text) for text in examples['text']]
    # 使用分词器对文本进行编码
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128
    )
    # 应用标签
    inputs['labels'] = examples['label']
    return inputs
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 移除不必要的列
tokenized_datasets = tokenized_datasets.remove_columns(['merged_text', 'comments', 'nonbug/bug_nonbug', 'nonbug/bug_bug', 'nonbug/bug_invalid'])


# PyTorch Tensor
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  
state_dict = torch.load('roberta_bug_report_model.pt')
robberta_state_dict = {k: v for k, v in state_dict.items() if k.startswith('roberta.')}
model.roberta.load_state_dict(robberta_state_dict, strict=False)

# 设置 参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
training_args = TrainingArguments(
    output_dir='./finetune_actrualbug_results',          # 训练输出目录
    overwrite_output_dir=True,                           # 覆盖输出目录
    num_train_epochs=16,                                  # 训练轮数
    per_device_train_batch_size=16,                       # 每个设备的训练批次大小
    per_device_eval_batch_size=16,                        # 每个设备的评估批次大小
    save_steps=10_000,                                   # 每10,000步保存一次模型
    save_total_limit=2,                                  # 最多保存2个检查点
    evaluation_strategy='epoch',                         # 每个epoch进行评估
    logging_dir='./logs',                                # 日志保存目录
    logging_steps=500,                                   # 每500步记录一次日志
)

#回测
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1) 
    labels = p.label_ids                       
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)

    return {
        'f1': f1,
        'accuracy': accuracy,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model('./finetuned_actrualbug_model')  
tokenizer.save_pretrained('./finetuned_actrualbug_model')  


predictions = trainer.predict(tokenized_datasets['test'])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")