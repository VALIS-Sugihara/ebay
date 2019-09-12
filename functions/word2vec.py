import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
from collections import defaultdict  # For word frequency
from google import Google
from rakuten import Rakuten

# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_sm')


from gensim.models.phrases import Phrases, Phraser
import multiprocessing

from gensim.models import Word2Vec


keywords = "leica"


def test(event, context):
    rakuten = Rakuten()
    google = Google()

    csv = "data/sample_ebay_leica.csv"
    data = pd.read_csv(csv)

    select_columns = ("itemId", "shortTitle", "title", "currentPrice", "currency")

    for c in data.columns:
        if c not in select_columns:
            data = data.drop(c, axis=1)

    rk_item_names = []
    rk_item_price = []

    # data = data.assign(
    #     rk_itemName=data.apply(lambda x: rakuten.get_items(rakuten.search(google.translate(text=x['shortTitle'])))[0].get("itemName", None), axis=1)
    # )
    # data = data.assign(
    #     rk_itemPrice=data.apply(lambda x: rakuten.get_items(rakuten.search(google.translate(text=x['shortTitle'])))[0].get("itemPrice", None), axis=1)
    # )

    for dt in data["shortTitle"]:
        # TODO:: ハンドリング
        try:
            dt = google.translate(text=dt)
            response = rakuten.search(keyword=dt)
            items = rakuten.get_items(response)
            rk_item_names.append(items[0].get("itemName", None))
            rk_item_price.append(items[0].get("itemPrice", 0))
        except:
            rk_item_names.append(None)
            rk_item_price.append(0)

    data.loc["rk_itemName"] = rk_item_names
    data.loc["rk_itemPrice"] = rk_item_price
    data.to_csv("./data/sample_rakuten_leica.csv")
    return

    title = data["title"][1]

    # 形態素解析
    doc = nlp(title)
    sentence = [d.text.lower() for d in doc]
    print(sentence)

    # キーワードから３単語分抽出
    target = int(sentence.index(keywords))
    length = target + 3
    print(sentence[target:length])

    # 翻訳
    google = Google()
    ja_keywords = " ".join(sentence[target:length])
    print(google.translate(text=ja_keywords))

    return

    for d in doc:
        print((d.text, d.pos_, d.dep_))

    print([(d.text, d.label_, spacy.explain(d.label_)) for d in doc.ents])

    return


    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data['title'])

    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    print(df_clean.head())
    print(df_clean.shape)


    sent = [row.split() for row in df_clean['clean']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)
    print(phrases)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    print(sentences)

    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    print(len(word_freq))
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])


def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)


def training(sentences):
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=20,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores-1)

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)


def frequent_title_in_category(df, category_column):
    from collections import Counter

    categories = df[category_column].value_counts().index
    # TODO:: eachClasses
    for category_name in categories:
        each_df = df[df[category_column] == category_name]
        words = " ".join(each_df["title"].to_list())
        ptn = r"[^a-zA-Z0-9\s]"
        words = re.sub(ptn, "", words.lower())
        doc = nlp(words)
        words = [token.text for token in doc]
        word_freq = Counter(words)
        common_words = word_freq.most_common(30)

        yield category_name, common_words

        # import matplotlib.pyplot as plt
        # words = [x for x, y in common_words]
        # counts = [y for x, y in common_words]
        # plt.bar(range(0, len(words)), counts, tick_label=words)
        # plt.show()
        #
        # print(category_name, common_words)


def analyze_category():
    # TEST::
    df = pd.read_csv("data/sample_ebay_leica.csv")
    frequency_list = list(frequent_title_in_category(df, "primaryCategory.categoryName"))

    select_columns = ("title", "primaryCategory.categoryId", "primaryCategory.categoryName")

    for c in df.columns:
        if c not in select_columns:
            df = df.drop(c, axis=1)

    print(df.head())

    titles = df["title"]
    categories = []
    for ttl in titles:
        top_score = 0
        top_category = None
        for lst in frequency_list[0:10]:
            score = 1  # 減点法
            category = lst[0]
            print("CHECK ...", category)
            # print("words ...", lst[1])
            total = np.sum([cnt for target, cnt in lst[1]])
            for target, cnt in lst[1]:
                if target.lower() not in ttl.lower().split():
                    score -= cnt / total
            print(score, top_score)
            if score > top_score:
                top_category = lst[0]
                top_score = score
        categories.append(top_category)

    df = df.assign(predictCategory=categories)

    df.to_csv("data/sample_ebay_predict.csv")

# test(True, True)

# df = pd.read_csv("data/sample_ebay_leica.csv")
# frequent_title_in_category(df, "primaryCategory.categoryName")
analyze_category()

# from sklearn import tree
