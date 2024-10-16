import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 加载数据 选择列 结合列 
df = pd.read_csv('bug_report.csv')
df = df[['merged_text', 'comments', 
         'impact_build/compilation error', 
         'impact_crash/exception', 
         'impact_operation failure', 
         'impact_others', 
         'impact_warning style error', 
         'impact_wrong output']]

df['text'] = df['merged_text'] + " " + df['comments']
df['text'] = df['text'].fillna('').astype(str)

#  建标签列
def create_labels(row):
    return [
        float(row['impact_build/compilation error']),
        float(row['impact_crash/exception']),
        float(row['impact_operation failure']),
        float(row['impact_others']),
        float(row['impact_warning style error']),
        float(row['impact_wrong output'])
    ]

df['labels'] = df.apply(create_labels, axis=1)

#随机分割 80% - 20% 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 转换为 Hugging Face 的 Dataset 格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 创建 DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

#分词器 数据预处理
tokenizer = RobertaTokenizer.from_pretrained('./roberta_bug_report_model')  # 指向您的分词器目录

def preprocess_function(examples):
    texts = [str(text) for text in examples['text']]
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128
    )
    # 应用多标签
    inputs['labels'] = [labels for labels in examples['labels']]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 移除不必要的列
tokenized_datasets = tokenized_datasets.remove_columns([
    'merged_text', 
    'comments', 
    'impact_build/compilation error', 
    'impact_crash/exception', 
    'impact_operation failure', 
    'impact_others', 
    'impact_warning style error', 
    'impact_wrong output'
])

# Tensor设置
tokenized_datasets.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'labels']
)

# 加载预训练模型 设置分类数 
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=6,
    problem_type="multi_label_classification"
)

try:
    state_dict = torch.load('./roberta_bug_report_model.pt', map_location=torch.device('cpu'), weights_only=True)
except TypeError:
    state_dict = torch.load('./roberta_bug_report_model.pt', map_location=torch.device('cpu'))
roberta_state_dict = {k: v for k, v in state_dict.items() if k.startswith('roberta.')}
model.roberta.load_state_dict(roberta_state_dict, strict=False)


# 设置设备 参数 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
training_args = TrainingArguments(
    output_dir='./finetune_impact_results',          
    overwrite_output_dir=True,                       # 覆盖输出目录 
    num_train_epochs=16,                              
    per_device_train_batch_size=16,                  # 训练批次
    per_device_eval_batch_size=8,                    # 评估批次
    save_strategy='epoch',                           # 保存策略设置为 'epoch'，以匹配评估策略
    save_total_limit=2,                              # 最多保存2个检查点
    evaluation_strategy='epoch',                     # 每个epoch进行评估
    logging_dir='./impact_logs',                     # 日志保存目录
    logging_steps=500,                               # 每500步记录一次日志
    learning_rate=3e-5,                              # 学习率
    load_best_model_at_end=True,                     # 在训练结束时加载最佳模型
    metric_for_best_model='f1',                      # 根据 F1 分数选择最佳模型
)

#回测
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions))  
    preds = (preds > 0.5).int().numpy()               
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
trainer.save_model('./finetuned_impact_model')  
tokenizer.save_pretrained('./finetuned_impact_model')  

predictions = trainer.predict(tokenized_datasets['test'])
preds = (torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5).int().numpy()
labels = predictions.label_ids
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")