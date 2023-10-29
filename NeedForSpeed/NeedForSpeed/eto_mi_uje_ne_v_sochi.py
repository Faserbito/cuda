import stanza

nlp = stanza.Pipeline('ru', processors='tokenize,ner,pos,lemma')

# # stanza.download('en')
# # stanza.download('ru')

# # text = """«Сегодня»: 27 октября 2023 года. 08:00 | Выпуск новостей | Новости НТВ

# # 00:00 В этом выпуске
# # 01:11 Космические возможности: что обсуждалось на встрече Владимира Путина с молодыми учеными.
# # 07:07 Силы США по приказу Джо Байдена нанесли удары по двум объектам в Сирии.
# # 08:49 Ракета попала в больницу в Египте у границы с Израилем: есть раненые.
# # 09:30 Группировка российских войск «Запад» отбила шесть атак штурмовых групп ВСУ.
# # 11:13 Стражи неба: российские зенитчики на Южнодонецком направлении уничтожают запасы украинских дронов с помощью ракетного комплекса «Тор».
# # 14:19 Спикер Палаты представителей США выступил за разделение помощи Израилю и Украине.
# # 15:57 Шаг из строя: как в Брюсселе становится все больше противников содержать киевский режим.
# # 19:27 В областях Центрального региона после снегопада объявили погодные предупреждения.

# # «Stupid Niggas» going in Seattle, November 15

# # Больше новостей — в Telegram. Подписывайтесь: https://t.me/ntvnews
# # Смотрите все выпуски «Сегодня» на RUTUBE: https://rutube.ru/plst/53542/"""
from work_with_excel import work

def stanza_ru(row, text):
    doc = nlp(text)
    for el in doc.sentences:        
        sp = []
        for ent in range(len(el.entities)):
            sp.append(el.entities[ent].type)
        print(sp)
        # work(row, sp)
            # if el.entities[ent].type not in ('O'):
            #     sp.append[el.entities[ent].type]
                # print(el.entities[ent].text, '\t', el.entities[ent].type)
                
            
# # stanza_ru(text)


import pandas as pd
data = pd.read_csv("ner_data_train.csv")
import json
df = data.copy()
df['entities'] = df['entities'].apply(lambda l: l.replace('\,', ',')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: l.replace('\\\\', '\\')if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: '[' + l + ']'if isinstance(l, str) else l)
df['entities'] = df['entities'].apply(lambda l: json.loads(l)if isinstance(l, str) else l)
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
    
from sklearn.model_selection import train_test_split
ner_data = [extract_labels(item) for i, item in df.iterrows()]
ner_train, ner_test = train_test_split(ner_data, test_size=0.2, random_state=1)

pd.options.display.max_colwidth = 300
pd.DataFrame(ner_train).sample(3)

label_list = sorted({label for item in ner_train for label in item['tags']})
if 'O' in label_list:
    label_list.remove('O')
    label_list = ['O'] + label_list

from datasets import Dataset, DatasetDict

ner_data = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(ner_train)),
    'test': Dataset.from_pandas(pd.DataFrame(ner_test))
})




# from work_with_excel import work

# print(len(ner_train))
for i in range(30):
    stanza_ru(i + 2, " ".join(ner_train[i]["tokens"]))
    # for j in range(len(i["tokens"])):
    #     work((i["tokens"][j], i["tags"][j]))
    #     print(f'{i["tokens"][j]}: {i["tags"][j]}\n')
    # print("------------------------------")




# #557570


# from spacy.lang.ru import Russian
# import spacy
# import random
# from spacy.training import Example

# nlp = spacy.load("ru_core_news_sm")
# # for i in range(10):
# #     print(" ".join(ner_train[i]["tokens"]).replace("=", "").replace("<", "").replace(">", ""))
# #     print("-----------------------------------------------")
# #     doc = nlp(" ".join(ner_train[i]["tokens"]).replace("=", "").replace("<", "").replace(">", ""))    

# TRAINING_DATA = [
#     (" НАЗВАНИЕ :  ТОП  5 хороших видеорегистраторов с двумя камерами . Что выбрать в 2022 ?  ОПИСАНИЕ :  5 . Roadgid Blick GPS :  LINK  4 . iBOX iCON LV WiFi Signature Dual :  LINK  3. 70 mai Dash Cam Pro Plus :  LINK  2 . Daocam Combo 2 CH :  LINK  1 . RoadGid CityGo 3 2 CH :  LINK  Мы Вконтакте :  LINK  Мы в Telegram :  LINK  Наш канал на Я . Дзен :  LINK  Мы в ТикТок :  LINK  /  AT  Сотрудничество и реклама :  AT  Прямая связь с автором :  AT  ✔ Нажми на колокольчик ! ✔ Ставь лайк , если нравится ! ✔ Общайся в комментариях PRO АВТО © 2017",
#     {"entities": [(85,92,"DATA"), (111, 129,"MODEL"), (142, 175,"MODEL"), (187, 212,"MODEL"), (225, 243,"MODEL"), (256, 278,"MODEL"), (290, 300,"ORGANIZATION"), (314, 323,"ORGANIZATION"), (345, 354,"ORGANIZATION"), (368, 375,"ORGANIZATION"), (534, 543,"ORGANIZATION"), (545, 550,"DATA")]}),
#     # {"entities": [(111, 129,"MODEL")]},
#     # {"entities": [(142, 175,"MODEL")]},
#     # {"entities": [(187, 212,"MODEL")]},
#     # {"entities": [(225, 243,"MODEL")]},
#     # {"entities": [(256, 278,"MODEL")]},
#     # {"entities": [(290, 300,"ORGANIZATION")]},
#     # {"entities": [(314, 323,"ORGANIZATION")]},
#     # {"entities": [(345, 354,"ORGANIZATION")]},
#     # {"entities": [(368, 375,"ORGANIZATION")]},
#     # {"entities": [(534, 543,"ORGANIZATION")]},
#     # {"entities": [(545, 550,"DATA")]}),
#     (" НАЗВАНИЕ :  День создания органов государственного пожарного надзора  ОПИСАНИЕ :  Огнеборцы России отметили профессиональный праздник — День создания органов государственного пожарного надзора . В этом году службе исполнилось 96 лет . О буднях и рабочих праздниках поговорили с заместителем начальника ГУ МЧС России по Северной Осетии Павелом Джанаевым .",
#     {"entities": [(34,77,"ORGANIZATION"), (100, 107,"LOCATION"), (310, 324,"ORGANIZATION"), (327, 343,"LOCATION"), (343, 361,"PERSON")]})
#     # {"entities": [(100, 107,"LOCATION")]},
#     # {"entities": [(310, 324,"ORGANIZATION")]},
#     # {"entities": [(327, 343,"LOCATION")]},
#     # {"entities": [(343, 361,"PERSON")]})
    
# ]
    
# nlp = Russian()

# for i in range(2):
#     random.shuffle(TRAINING_DATA)
#     for batch in spacy.util.minibatch(TRAINING_DATA, 2):
#         texts = [text for text, annotation in batch]
#         annotations = [annotation for text, annotation in batch]
#         example = Example.from_dict(nlp.make_doc(texts), annotations)
#         nlp.update([example])
#         # nlp.update(texts, annotations)
        
# nlp.to_disk("model")
    
    
    
     
     
#     # Другие примеры...

# nlp = English()

# for i in range(10):
#     random.shuffle(TRAINING_DATA)
#     for batch in spacy.util.minibatch(TRAINING_DATA):
#         texts = [text for text, annotation in batch]
#         annotations = [annotation for text, annotation in batch]
#         nlp.update(texts, annotations)
        
# nlp.to_disk("model")
# doc = nlp("Космические возможности: что обсуждалось на встрече Владимира Путина с Владимиром Зеленским и Геннадием Цидармяном в здании Вагнер-центра")
# for ent in doc.ents:
#     print(ent.text, ent.label_)

