import json
from tqdm import tqdm
import codecs

train = []
chars = {}

with open('data/train_data_all.json', 'r', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        for doc in a['document']:
            for i in a['qas']:
                tmp = []
                for j in i:
                    for k in j['answers']:
                        tmp.append(k['text'])
            train.append(
                {
                    'text': doc['text'],
                    'spo_list': tmp
                }
            )
        for c in doc['text']:
            chars[c] = chars.get(c, 0) + 1

# 整理之后的json文档
# with codecs.open('data/train_data_me.json', 'w', encoding='utf-8') as f:
#     json.dump(train, f, indent=4, ensure_ascii=False)

# 位置特征
with codecs.open('data/all_chars_me.json', 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= 2}
    id2char = {i+2:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

# all_50_schemas = set()
#
# with open('data/all_50_schemas.json') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         all_50_schemas.add(a['predicate'])
# id2predicate = {i:j for i,j in enumerate(all_50_schemas)}
# predicate2id = {j:i for i,j in id2predicate.items()}
#
# with codecs.open('data/all_50_schemas_me.json', 'w', encoding='utf-8') as f:
#     json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)