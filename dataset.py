import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re


file_info = [
    (r'bugreport_crawler\tc_bugreports\tc.json', 'tc'),
    (r'bugreport_crawler\tvm_bugreports\tvm.json', 'tvm'),
    (r'bugreport_crawler\plaidml_bugreports\plaidml.json', 'plaidml'),
    (r'bugreport_crawler\ngraph_bugreports\ngraph.json', 'ngraph'),
    (r'bugreport_crawler\glow_bugreports\glow.json', 'glow')
]

merged_data = []


for file_name, category in file_info:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            item['summary'] += f" The framework I use is {category}."
        merged_data.extend(data)

# 将合并后的数据写入新的 JSON 文件
with open('all_bug_report.json', 'w', encoding='utf-8') as merged_file:
    json.dump(merged_data, merged_file, ensure_ascii=False, indent=4)

print("所有文件已成功合并到 all_bug_report.json")



# 高频词提取
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
with open('all_bug_report.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 统计数据条数
total_entries = len(data)


def clean_and_tokenize(text):
    # 移除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转小写并分词
    words = text.lower().split()
    # 去除停用词和单字母词
    return [word for word in words if word not in stop_words and len(word) > 1]

word_counter = Counter()

# 遍历数据进行统计
for item in data:
    if 'title' in item:
        word_counter.update(clean_and_tokenize(item['title']))
    if 'summary' in item:
        word_counter.update(clean_and_tokenize(item['summary']))
    if 'comments' in item:
        word_counter.update(clean_and_tokenize(item['comments']))

# 频率阈值
frequency_threshold = 0.0005  # 1%

total_words = sum(word_counter.values())
frequent_words = [word for word, count in word_counter.items() if count / total_words > frequency_threshold]


with open('frequent_words.json', 'w', encoding='utf-8') as output_file:
    json.dump(frequent_words, output_file, ensure_ascii=False, indent=4)

print(f"数据条数: {total_entries}")
print("高频词统计已完成并存储在 frequent_words.json")