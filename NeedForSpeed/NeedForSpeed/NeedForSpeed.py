# считаем данные
import pandas as pd
data = pd.read_csv("ner_data_train.csv")
data.head(5)
# данные спарсены с Толоки, поэтому могут иметь проблемы с символами и их нужно избежать, 
# удалить лишние '\' например, преобразовать из str в список dict-ов
import json
df = data.copy()
df['entities'] = df['entities'].apply(lambda l: l.replace('\,', ',')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: l.replace('\\\\', '\\')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: '[' + l + ']'if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: json.loads(l)if isinstance(l, str) else l)
df.head(3)
# Теперь из наших данных нам нужно извлечь для каждого слова (токена) его тег (label) из разметки, чтобы потом предать в модель классификации токенов
from razdel import tokenize

def extract_labels(item):
    
    # воспользуемся удобным токенайзером из библиотеки razdel, 
    # она помимо разбиения на слова, сохраняет важные для нас числа - начало и конец слова в токенах
    
    raw_toks = list(tokenize(item['video_info']))
    words = [tok.text for tok in raw_toks]
    # присвоим для начала каждому слову тег 'О' - тег, означающий отсутствие NER-а
    word_labels = ['O'] * len(raw_toks)
    char2word = [None] * len(item['video_info'])
    # так как NER можем состаять из нескольких слов, то нам нужно сохранить эту инфорцию
    for i, word in enumerate(raw_toks):
        char2word[word.start:word.stop] = [i] * len(word.text)

    labels = item['entities']
    if isinstance(labels, dict):
        labels = [labels]
    if labels is not None:
        for e in labels:
            if e['label'] != 'не найдено':
                e_words = sorted({idx for idx in char2word[e['offset']:e['offset']+e['length']] if idx is not None})
                if e_words:
                    word_labels[e_words[0]] = 'B-' + e['label']
                    for idx in e_words[1:]:
                        word_labels[idx] = 'I-' + e['label']
                else:
                    continue
            else:
                continue
        return {'tokens': words, 'tags': word_labels}
    else: return {'tokens': words, 'tags': word_labels}
print(extract_labels(df.iloc[0]))
from sklearn.model_selection import train_test_split
ner_data = [extract_labels(item) for i, item in df.iterrows()]
ner_train, ner_test = train_test_split(ner_data, test_size=0.2, random_state=1)
import pandas as pd
pd.options.display.max_colwidth = 300
pd.DataFrame(ner_train).sample(3)
label_list = sorted({label for item in ner_train for label in item['tags']})
if 'O' in label_list:
    label_list.remove('O')
    label_list = ['O'] + label_list
label_list
from datasets import Dataset, DatasetDict
ner_data = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(ner_train)),
    'test': Dataset.from_pandas(pd.DataFrame(ner_test))
})
ner_data
from transformers import AutoTokenizer 
from datasets import load_dataset, load_metric

model_checkpoint = "cointegrated/rubert-tiny"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device='cpu')
example = ner_train[5]
print(example["tokens"])
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)
len(example["tags"]), len(tokenized_input["input_ids"])
print(tokenized_input.word_ids())
word_ids = tokenized_input.word_ids()
aligned_labels = [-100 if i is None else example["tags"][i] for i in word_ids]
print(len(aligned_labels), len(tokenized_input["input_ids"]))
def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids = [label_list.index(idx) if isinstance(idx, str) else idx for idx in label_ids]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
tokenize_and_align_labels(ner_data['train'][1:2])
tokenized_datasets = ner_data.map(tokenize_and_align_labels, batched=True)
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
model.config.id2label = dict(enumerate(label_list))
model.config.label2id = {v: k for k, v in model.config.id2label.items()}
# Специальный объект для удобного формирования батчей
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")
example = ner_train[4]
labels = example['tags']
metric.compute(predictions=[labels], references=[labels])
args = TrainingArguments(
    "ner",
    evaluation_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy='no',
    report_to='none',
)
# что мы видим без дообучения модели
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()
import logging
from transformers.trainer import logger as noisy_logger
noisy_logger.setLevel(logging.WARNING)
# Для дообучения берта можно эксперементировать с заморозкой/разморозкой разных слоев, здесь мы оставим все слои размороженными 
# Для быстроты обучения можно заморозить всю бертовую часть, кроме классификатора, но тогда качесвто будет похуже
for param in model.parameters():
    param.requires_grad = True
    args = TrainingArguments(
    "ner",
    evaluation_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy='no',
    report_to='none',
)
    trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# trainer.train()
# trainer.evaluate()
# Посчитаем метрики на отложенном датасете

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
from sklearn.metrics import confusion_matrix
import pandas as pd
cm = pd.DataFrame(
    confusion_matrix(sum(true_labels, []), sum(true_predictions, []), labels=label_list),
    index=label_list,
    columns=label_list
)
cm
model.save_pretrained('ner_bert.bin')
tokenizer.save_pretrained('ner_bert.bin')
# text = ' '.join(ner_train[25]['tokens'])
text = ner_train[25]['tokens']
import torch
from transformers import pipeline

pipe = pipeline(model=model, tokenizer=tokenizer, task='ner', aggregation_strategy='average', device='cpu')

def predict_ner(text, tokenizer, model, pipe, verbose=True):
    tokens = tokenizer(text, truncation=True, is_split_into_words=True, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    
    with torch.no_grad():
        pred = model(**tokens)
    # print(pred.logits.shape)
    indices = pred.logits.argmax(dim=-1)[0].cpu().numpy()
    token_text = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    labels = []
    for t, idx in zip(token_text, indices):
        if '##' not in t:
            labels.append(label_list[idx])
        if verbose:    
            print(f'{t:15s} {label_list[idx]:10s}')
    return (text, pipe(text), labels)

print(predict_ner(text, tokenizer, model, pipe))
from tqdm.notebook import tqdm

submission = pd.DataFrame(columns=[['video_info', 'entities_prediction']])
submission['entities_prediction'] = submission['entities_prediction'].astype('object')
def sample_submission(text, tokenizer, model, pipe, submission):
    for i, elem in tqdm(enumerate(ner_test)):
        _, _, labels = predict_ner(elem['tokens'], tokenizer, model, pipe, verbose=False)
        submission.loc[i, 'video_info'] = elem

        submission.loc[i, 'entities_prediction'] = [[label] for label in labels]
    return submission
result = sample_submission(text, tokenizer, model, pipe, submission)
print(result)
len(ner_test)