import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy
import re
from collections import defaultdict  # For word frequency

from google import Google
from rakuten import Rakuten

import redis
# Redis に接続します
r = redis.Redis(host='localhost', port=6379, db=0)

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


def frequent_title_in_category(df, column_names):
    """
    df のタイトル
    :param df:
    :param column_names:
    :return:
    """
    from collections import Counter
    from dictionary import FrequentDictionary
    fd = FrequentDictionary()

    categories = df[column_names[1]].value_counts().index
    print(categories)
    # TODO:: eachClasses
    for category_name in categories:
        each_df = df[df[column_names[1]] == category_name]
        all_texts = each_df[column_names[0]].to_list()
        words = " ".join(all_texts)
        ptn = r"[^a-zA-Z0-9\s]"
        words = re.sub(ptn, "", words.lower())
        doc = nlp(words)

        # 翻訳リストに追加 & return
        words = [fd.register(token.text.strip()) for token in doc]
        word_freq = Counter(words)
        common_words = word_freq.most_common(30)

        # 頻出リストに追加
        print("================")
        print(category_name, common_words)
        print("================")
        # return
        fd.set_frequency(brand="ebay", key=category_name, value=common_words)

        yield (category_name, common_words,)

        # import matplotlib.pyplot as plt
        # words = [x for x, y in common_words]
        # counts = [y for x, y in common_words]
        # plt.bar(range(0, len(words)), counts, tick_label=words)
        # plt.show()
        #
        # print(category_name, common_words)


def analyze_category(brand_name="ebay"):
    KEYWORDS = "nikon"  # TEST

    # BRAND, TITLE, CATEGORY_ID, CATEGORY_NAME
    column_names = {
        "ebay": ("title", "primaryCategory.categoryName", "primaryCategory.categoryId",),
        "yahoo": ("Title", "CategoryId",)
    }

    # TEST::
    df = pd.read_csv("data/sample_%s_%s.csv" % (brand_name, KEYWORDS,))
    frequency_list = list(frequent_title_in_category(df, column_names[brand_name]))

    # for c in df.columns:
    #     if c not in select_columns:
    #         df = df.drop(c, axis=1)

    # TEST::
    # brand_name="yahoo"
    # df = pd.read_csv("data/sample_%s_%s.csv" % ("yahoo", KEYWORDS,))
    from dictionary import Dictionary
    d = Dictionary()

    titles = df[column_names[brand_name][0]]
    categories = []
    scores = []
    for ttl in titles:
        # en に翻訳
        # ttl = d.register(ttl)
        # ttl = ttl.decode() if isinstance(ttl, bytes) else ttl
        top_score = 0
        top_category = None
        for lst in frequency_list[0:10]:
            score = 1  # 減点法
            category = lst[0]
            # print("words ...", lst[1])
            total = np.sum([cnt for target, cnt in lst[1]])
            for target, cnt in lst[1]:
                target = target.decode() if isinstance(target, bytes) else target

                doc = nlp(ttl)
                # 翻訳リストに追加 & return
                words = [token.text.strip() for token in doc]

                # if target.lower() not in re.split(r"[,\s.]", ttl.lower()):
                if target.lower() not in words:
                    score -= cnt / total
            if score > top_score:
                print(top_score, top_category)
                top_category = lst[0]
                top_score = score
        print(ttl, top_category)
        categories.append(top_category)
        scores.append(top_score)

    df = df.assign(predictCategory=categories)
    df = df.assign(score=scores)

    df.to_csv("data/sample_%s_predict.csv" % (brand_name,))


def each_category_and_score(df, brand_name="ebay"):
    KEYWORDS = "nikon"  # TEST

    # BRAND, TITLE, CATEGORY_ID, CATEGORY_NAME
    column_names = {
        "ebay": ("title", "primaryCategory.categoryName", "primaryCategory.categoryId",),
        "yahoo": ("Title", "CategoryId",)
    }

    frequency_list = list(frequent_title_in_category(df, column_names[brand_name]))

    # TEST::
    titles = df[column_names[brand_name][0]]

    # 初期化
    categories = {}
    for lst in frequency_list[0:10]:
        categories[lst[0]] = []
    for ttl in titles:
        top_score = 0
        top_category = None
        for lst in frequency_list[0:10]:
            score = 1  # 減点法
            category = lst[0]
            # print("words ...", lst[1])
            total = np.sum([cnt for target, cnt in lst[1]])
            for target, cnt in lst[1]:
                target = target.decode() if isinstance(target, bytes) else target

                doc = nlp(ttl)
                # 翻訳リストに追加 & return
                words = [token.text.strip() for token in doc]

                # if target.lower() not in re.split(r"[,\s.]", ttl.lower()):
                if target.lower() not in words:
                    score -= cnt / total

            categories[category].append(score)

            if score > top_score:
                print(top_score, top_category)
                top_category = lst[0]
                top_score = score

    print(categories)
    for category, score in categories.items():
        df[category] = score

    print(df.head())

    return df





# test(True, True)



# df = pd.read_csv("data/sample_ebay_leica.csv")
# frequent_title_in_category(df, "primaryCategory.categoryName")

# analyze_category(brand_name="ebay")



# import statsmodels.api as sm
# # statsmodelを利用してロジスティック回帰のモデルを構築
# mushroom2 = sm.add_constant(df)
# logit = sm.Logit(mushroom2['class'], mushroom2[['const','bruises_t']])
# result = logit.fit()
#
# # 訓練ずみモデルの詳細確認
# result.summary()


def logistic_regression(df):
    ## データ処理と可視化のためのライブラリー
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # 機械学習ライブラリ「Scikit-learn」のインポート
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    df2 = pd.read_csv("data/sample_ebay_predict.csv")
    predicts = df2[["predictCategory"]]
    df["predictCategory"] = predicts
    df = df.assign(result=lambda x: x["predictCategory"] == x["primaryCategory.categoryName"])
    # 文字列から数値へ変換
    labelencoder = LabelEncoder()
    df['result'] = labelencoder.fit_transform(df['result'])

    columns = ['result', 'Lenses', 'Film Cameras',
       'Viewfinders & Eyecups', 'Digital Cameras', 'Flashes', 'Lens Hoods',
       'Lens Caps', 'Battery Grips', 'Lens Adapters, Mounts & Tubes',
       'Cases, Bags & Covers']

    df = df[columns]

    # 訓練データ（80%）とテストデータ（20%)へスプリット
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    # 特徴量（x）とターゲット（y）へ切り分け
    X_train = train_set.drop('result',axis=1).fillna(0)
    y_train = train_set['result'].copy().fillna(0)

    X_test = test_set.drop('result',axis=1).fillna(0)
    y_test = test_set['result'].copy().fillna(0)

    # RFEを使って特徴選択を行います
    logreg = LogisticRegression()
    rfe = RFE(logreg, 5, verbose=1)
    rfe = rfe.fit(X_train, y_train)

    # 選択した特徴量を切り分けます
    X_train = X_train[X_train.columns[rfe.support_]]
    X_test = X_test[X_test.columns[rfe.support_]]

    # データフレームの確認
    print(X_train.head())

    # 訓練データを使ってモデルの訓練
    logclassifier = LogisticRegression()
    logclassifier.fit(X_train, y_train)

    # 訓練データの予測
    y_pred = logclassifier.predict(X_train)

    # 混同行列で訓練データの予測結果を評価
    cnf_matrix = confusion_matrix(y_train, y_pred)
    print(cnf_matrix)

    # 正解率を算出
    print(accuracy_score(y_train, y_pred))

    # テストデータの予測
    y_pred_test = logclassifier.predict(X_test)

    # 混同行列（テストデータ）
    cnf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print(cnf_matrix_test)

    # 正解率（テストデータ）
    print(accuracy_score(y_test, y_pred_test))


df = pd.read_csv("data/category_ebay.csv", encoding='cp932')
logistic_regression(df)



def test():
    # 機械学習ライブラリ「Scikit-learn」のインポート
    from sklearn.preprocessing import LabelEncoder
    # 新しい書き方
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv("data/sample_ebay_predict.csv")
    df = df.assign(result=lambda x: x["predictCategory"] == x["primaryCategory.categoryName"])
    df = df[["score", "result"]]
    # 文字列から数値へ変換
    labelencoder = LabelEncoder()
    df['result'] = labelencoder.fit_transform(df['result'])
    print(df.head())
    print(df["result"].value_counts())
    # 訓練データとテストデータへスプリット
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    # 訓練データの特徴量とターゲットを切り分ける
    X_train = train_set.drop('result', axis=1)
    y_train = train_set['result'].copy()

    # テストデータの特徴量とターゲットを切り分ける
    X_test = test_set.drop('result', axis=1)
    y_test = test_set['result'].copy()

    # 訓練データをロジスティック回帰のモデルへ訓練
    logclassifier = LogisticRegression()
    logclassifier.fit(X_train, y_train)

    # 訓練ずみモデルを使って訓練データから予測する
    y_pred = logclassifier.predict(X_train)

    # 混同行列を作成
    cnf_matrix = confusion_matrix(y_train, y_pred)
    print(cnf_matrix)
    # 正解率を計算する
    print(accuracy_score(y_train, y_pred))

    # 訓練ずみモデルからテストデータを使って予測
    y_pred_test = logclassifier.predict(X_test)
    # 混同行列を作成
    cnf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print(cnf_matrix_test)
    # 正解率を計算する
    print(accuracy_score(y_test, y_pred_test))
