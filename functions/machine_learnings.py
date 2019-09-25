import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import pickle
import re
from collections import defaultdict  # For word frequency
from google import Google
from rakuten import Rakuten
# ハイパーパラメータチューニング
from sklearn.model_selection import GridSearchCV
import redis
# Redis に接続します
r = redis.Redis(host='localhost', port=6379, db=0)

# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_sm')

from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec

keywords = "leica"

TODAY = datetime.datetime.now().date().strftime("%Y%m%d")


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

        # 名詞、固有名詞、数字以外を除去
        words = []
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN", "NUM",):
                words.append(token.text)

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
        "yahoo": ("en_Title", "CategoryId",)
    }

    # TEST::
    df = pd.read_csv("data/sample_%s_%s.csv" % (brand_name, KEYWORDS,))
    frequency_list = list(frequent_title_in_category(df, column_names[brand_name]))

    # for c in df.columns:
    #     if c not in select_columns:
    #         df = df.drop(c, axis=1)

    # TEST::
    brand_name="yahoo"
    df = pd.read_csv("data/svn_%s_%s_en.csv" % ("yahoo", KEYWORDS,))
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
                top_category = lst[0]
                top_score = score
                print(top_score, top_category)
        print(ttl, top_category)
        categories.append(top_category)
        scores.append(top_score)

    df = df.assign(predictCategory=categories)
    df = df.assign(score=scores)

    df.to_csv("data/%s_predict_%s.csv" % (brand_name, TODAY,))


def each_category_and_score(df, brand_name="ebay"):
    KEYWORDS = "nikon"  # TEST

    # BRAND, TITLE, CATEGORY_ID, CATEGORY_NAME
    column_names = {
        "ebay": ("title", "primaryCategory.categoryName", "primaryCategory.categoryId",),
        "yahoo": ("en_Title", "CategoryId",)
    }

    frequency_df = pd.read_csv("data/ebay_detail_nikon_model_20190919.csv")
    frequency_list = list(frequent_title_in_category(frequency_df, column_names[brand_name]))

    # TEST::
    # brand_name = "yahoo"
    titles = df[column_names[brand_name][0]]

    # 初期化
    categories = {"predict_category": []}
    for lst in frequency_list[0:10]:
        categories[lst[0]] = []
    for ttl in titles:
        top_score = 0
        top_category = None
        for lst in frequency_list[0:10]:
            score = 1  # 減点法
            score = 0  # 加点法
            category = lst[0]
            # print("words ...", lst[1])
            total = np.sum([cnt for target, cnt in lst[1]])
            for target, cnt in lst[1]:
                target = target.decode() if isinstance(target, bytes) else target

                doc = nlp(ttl)
                # 翻訳リストに追加 & return
                words = [token.text.strip() for token in doc]

                # if target.lower() not in re.split(r"[,\s.]", ttl.lower()):
                # 減点法
                # if target.lower() not in words:
                #     score -= cnt / total
                # 加点法
                if target.lower() in words:
                    score += cnt / total

            categories[category].append(score)

            if score > top_score:
                # print(top_score, top_category)
                top_category = lst[0]
                top_score = score
        print(ttl, top_category)
        categories["predict_category"].append(top_category)

    print(categories)
    for category, score in categories.items():
        df[category] = score

    print(df.head())

    return df


def each_word_and_count(df, brand_name="ebay"):
    KEYWORDS = "nikon"  # TEST

    # BRAND, TITLE, CATEGORY_ID, CATEGORY_NAME
    column_names = {
        "ebay": ("title", "primaryCategory.categoryName", "primaryCategory.categoryId",),
        "yahoo": ("en_Title", "CategoryId",)
    }

    frequency_df = pd.read_csv("data/ebay_detail_nikon_model_%s.csv" % (TODAY,))
    frequency_list = list(frequent_title_in_category(frequency_df, column_names[brand_name]))

    # TEST::
    # brand_name = "yahoo"
    titles = df[column_names[brand_name][0]]

    # 初期化
    categories = {"predict_category": []}
    for lst in frequency_list[0:10]:
        # categories[lst[0]] = []
        for target, cnt in lst[1]:
            target = target.decode() if isinstance(target, bytes) else target
            categories[lst[0]+"_"+target.lower()] = []
    for ttl in titles:
        top_score = 0
        top_category = None
        for lst in frequency_list[0:10]:
            score = 1  # 減点法
            # score = 0  # 加点法
            category = lst[0]
            # print("words ...", lst[1])
            total = np.sum([cnt for target, cnt in lst[1]])
            for target, cnt in lst[1]:
                target = target.decode() if isinstance(target, bytes) else target

                doc = nlp(ttl)
                # 翻訳リストに追加 & return
                words = [token.text.strip() for token in doc]

                # if target.lower() not in re.split(r"[,\s.]", ttl.lower()):
                # 減点法
                if target.lower() not in words:
                    score -= cnt / total
                    categories["%s_%s" % (category, target.lower(),)].append(0)
                else:
                    categories["%s_%s" % (category, target.lower(), )].append(1)

                # 加点法
                # if target.lower() in words:
                #     score += cnt / total
                #     categories["%s_%s" % (category, target.lower(), )].append(1)
                # else:
                #     categories["%s_%s" % (category, target.lower(), )].append(0)

            # categories[category].append(score)

            if score > top_score:
                # print(top_score, top_category)
                top_category = lst[0]
                top_score = score
        print(ttl, top_category)
        categories["predict_category"].append(top_category)

    print(categories)
    for category, score in categories.items():
        df[category] = score

    print(df.head())

    return df


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

    # df2 = pd.read_csv("data/sample_ebay_predict.csv")
    # predicts = df2[["predictCategory"]]
    # df["predictCategory"] = predicts
    # df = df.assign(result=lambda x: x["predictCategory"] == x["primaryCategory.categoryName"])
    # 文字列から数値へ変換
    # labelencoder = LabelEncoder()
    # df['result'] = labelencoder.fit_transform(df['result'])

    # 特徴量をダミー変数化
    dfcol = df.columns[17:]
    print(dfcol)
    df1 = df[dfcol]
    print(df1.head())
    # df2 = pd.get_dummies(df["primaryCategory.categoryName"], prefix='primaryCategory', drop_first=True)
    # print(df2.head())
    # return
    # df2 = pd.get_dummies(df, columns=df[dfcol], drop_first=True)
    # print(df2.describe())
    # ターゲットの値を文字列から数値へ変換
    labelencoder = LabelEncoder()
    df['primaryCategory.categoryName'] = labelencoder.fit_transform(df['primaryCategory.categoryName'])
    df1["primaryCategory"] = df['primaryCategory.categoryName']
    df = df1
    print(df1.head())
    print(df1.tail())

    # columns = ['result', 'Lenses', 'Film Cameras',
    #    'Viewfinders & Eyecups', 'Digital Cameras', 'Flashes', 'Lens Hoods',
    #    'Lens Caps', 'Battery Grips', 'Lens Adapters, Mounts & Tubes',
    #    'Cases, Bags & Covers']

    # df = df[columns]

    # 訓練データ（80%）とテストデータ（20%)へスプリット
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    # 特徴量（x）とターゲット（y）へ切り分け
    X_train = train_set.drop('primaryCategory',axis=1).fillna(0)
    y_train = train_set['primaryCategory'].copy().fillna(0)

    X_test = test_set.drop('primaryCategory',axis=1).fillna(0)
    y_test = test_set['primaryCategory'].copy().fillna(0)

    # RFEを使って特徴選択を行います
    logreg = LogisticRegression()
    rfe = RFE(logreg, 50, verbose=1)
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


# df = pd.read_csv("data/ebay_categories_20190920.csv")
# logistic_regression(df)


def svm(df, grid_search_flg=True):
    # ライブラリのインポート
    import pandas as pd
    import sklearn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # df2 = pd.read_csv("data/sample_ebay_predict.csv")
    # predicts = df2[["predictCategory"]]
    # df["predictCategory"] = predicts
    # df = df.assign(result=lambda x: x["predict_category"] == x["primaryCategory.categoryName"])

    # df_origin = df

    # 文字列から数値へ変換
    # predict_types = df["predict_category"].value_counts().index.to_list()

    # labelencoder = LabelEncoder()
    # df = df.fillna("")
    # dummies = pd.get_dummies(df["predict_category"], prefix="predict_category")
    # print(dummies.head())
    # print(dummies.columns)
    # df = pd.merge(df, dummies, right_index=True, left_index=True)
    # df = df.assign(result=df.apply(lambda x: predict_types.index(x['predict_category']), axis=1))
    # results = df["result"]
    # indexes = df.index.to_list()
    # print(indexes)

    # # 文字列から数値へ変換
    # # labelencoder = LabelEncoder()
    # # df['result'] = labelencoder.fit_transform(df['result'])
    #
    # columns = [
    #     "result",
    # ] + indexes
    #
    # df = df[columns]
    #
    # print(df.head())
    #
    # df2 = pd.read_csv("data/yahoo_categories_20190919.csv")
    # # labelencoder = LabelEncoder()
    # df2 = df2.fillna("Lenses")
    # df2 = df2.assign(result=df2.apply(lambda x: indexes.index(x['predict_category']), axis=1))
    # df2 = df2[columns]

    # 特徴量をダミー変数化
    dfcol = df.columns[17:]
    print(dfcol)
    df1 = df[dfcol]
    print(df1.head())
    # df2 = pd.get_dummies(df["primaryCategory.categoryName"], prefix='primaryCategory', drop_first=True)
    # print(df2.head())
    # return
    # df2 = pd.get_dummies(df, columns=df[dfcol], drop_first=True)
    # print(df2.describe())
    # ターゲットの値を文字列から数値へ変換
    labelencoder = LabelEncoder()
    df1['primaryCategory'] = labelencoder.fit_transform(df['primaryCategory.categoryName'])
    # df1["primaryCategory"] = df['primaryCategory.categoryName']
    # df = df1
    print(df1.head())
    print(df1.tail())
    print(df1.columns.to_list())

    # 訓練データ/テストデータの分割
    train_set, test_set = train_test_split(df1, test_size = 0.2, random_state = 42)
    # train_set2, test_set2 = train_test_split(df2, test_size = 0.2, random_state = 42)

    # 訓練データの特徴量/ターゲットの切り分け
    X_train = train_set.drop('primaryCategory', axis=1).fillna(0)
    y_train = train_set['primaryCategory'].copy().fillna(0)

    # テストデータの特徴量/ターゲットの切り分け
    X_test = test_set.drop('primaryCategory', axis=1).fillna(0)
    y_test = test_set['primaryCategory'].copy().fillna(0)

    # Scikit-learnを利用して特徴量の正規化を行う
    # StandardScaler = StandardScaler()
    # X_train_norm = StandardScaler.fit_transform(X_train)
    # X_test_norm = StandardScaler.fit_transform(X_test)

    # 最初の5行を表示
    print(X_train.head())
    print(X_test.head())

    # 正規化後のsepal length(cm）の特徴量の平均
    # print(X_train_norm[:, 0].mean())

    # 正規化後のsepal length(cm）の特徴量の標準偏差
    # print(X_train_norm[:, 0].std())

    if grid_search_flg is False:
        # SVMのモデル訓練
        clf = svm.SVC()
        clf.fit(X_train, y_train)

        # 訓練データを使って予測
        y_pred_train = clf.predict(X_train)
        print(y_pred_train)

        # テストデータを使って予測
        y_pred_test = clf.predict(X_test)
        print(y_pred_test)

        # 訓練データの混同行列
        print(confusion_matrix(y_train, y_pred_train))

        # 訓練データの正解率
        print(accuracy_score(y_train, y_pred_train))

        # テストデータの混同行列
        print(confusion_matrix(y_test, y_pred_test))

        # テストデータでの正解率
        print(accuracy_score(y_test, y_pred_test))
    else:
        # SVMのモデル訓練
        clf = svm.SVC()

        # グリッドサーチで検証するSVMのハイパーパラメータの設定
        param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10],
                       'gamma': [.0001, .001, .01, .1, .2, .5, 1, 10],
                       'kernel': ['poly']},
                      {'C': [.001, .01, .1, 1, 10, 100, 1000],
                       'gamma': [.0001, .001, .01, .1, .2, .5, 1, 10],
                       'kernel': ['linear']},
                      {'C': [.001, .01, .1, 1, 10, 100, 1000],
                       'gamma': [.0001, .001, .01, .1, .2, .5, 1, 10],
                       'kernel': ['rbf']}
                      ]

        # グリッドサーチの構築
        grid_searchlog = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

        # グリッドサーチの実行
        grid_searchlog.fit(X_train, y_train)

        GridSearchCV(cv=5, error_score='raise',
                     estimator=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                                   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                                   tol=0.001, verbose=False),
                                   iid=True, n_jobs=1,
                     param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10],
                                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 10],
                                  'kernel': ['poly']}, {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                                        'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 10],
                                                        'kernel': ['linear']},
                                 {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 10],
                                  'kernel': ['rbf']}],
                     pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
                     scoring='accuracy', verbose=0)

        # 最適なハイパーパラメータの値を確認
        print(grid_searchlog.best_params_)

        # グリッドサーチで算出した最適なハイパーパラメータの値
        optimised_svm = grid_searchlog.best_estimator_
        print(optimised_svm)

        # 最適化されたSVMモデルと訓練データの予測
        train_y_pred_opt = optimised_svm.predict(X_train)
        print(train_y_pred_opt)

        # 訓練データの正解率（ハイパーパラメータチューニング実施後）
        print(accuracy_score(y_train, train_y_pred_opt))

        # 最適化されたSVMモデルとテストデータの予測
        test_y_pred_opt = optimised_svm.predict(X_test)
        print(test_y_pred_opt)

        # テストセットの予測結果とIDをデータフレーム型に変更
        my_solution_opt = pd.DataFrame(test_y_pred_opt, X_test.index, columns=['type'])
        print(my_solution_opt)

        # CSVファイルとして書き出し
        my_solution_opt.to_csv('./data/2nd_submit.csv', index_label=['id'])

        # モデルを保存する
        filename = 'models/finalized_grid_search.sav'
        pickle.dump(grid_searchlog, open(filename, 'wb'))

        # 保存したモデルをロードする
        # loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(X_test, Y_test)


df = pd.read_csv("data/ebay_categories_20190920.csv")
svm(df)


def k_means(df):
    # ライブラリのインポート
    import pandas as pd
    import sklearn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans

    indexes = [
        "Lenses",
        "Film Cameras",
        "Viewfinders & Eyecups",
        "Digital Cameras",
        "Flashes",
        "Lens Hoods",
        "Battery Grips",
        "Lens Adapters, Mounts & Tubes",
        "Film Backs & Holders",
        "Straps & Hand Grips",
    ]
    df = df.fillna("Lenses")

    df = df.assign(result=df.apply(lambda x: indexes.index(x['predict_category']), axis=1))

    n_clusters = len(indexes)

    sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                           tol=0.0001, precompute_distances='auto', verbose=0,
                           random_state=None, copy_x=True, n_jobs=1)

    columns = [
        # "result",
        "Lenses",
        "Film Cameras",
        "Viewfinders & Eyecups",
        "Digital Cameras",
        "Flashes",
        "Lens Hoods",
        "Battery Grips",
        "Lens Adapters, Mounts & Tubes",
        "Film Backs & Holders",
        "Straps & Hand Grips",
    ]

    df = df[columns]

    print(df.head())

    array = np.array([
        # df['result'].tolist(),
        df['Lenses'].tolist(),
        df['Film Cameras'].tolist(),
        df['Viewfinders & Eyecups'].tolist(),
        df['Digital Cameras'].tolist(),
        df['Flashes'].tolist(),
        df['Lens Hoods'].tolist(),
        df['Lens Hoods'].tolist(),
        df['Battery Grips'].tolist(),
        df['Lens Adapters, Mounts & Tubes'].tolist(),
        df['Film Backs & Holders'].tolist(),
        df['Straps & Hand Grips'].tolist(),
    ], np.float64)

    array = array.T
    print(array)

    pred = KMeans(n_clusters=n_clusters).fit_predict(array)
    print(pred)
    df["cluster_id"] = pred
    print(df.head())
    print(df["cluster_id"].value_counts())


    # 可視化（積み上げ棒グラフ）
    # import matplotlib.pyplot as plt

    clusterinfo = pd.DataFrame()
    for i in range(n_clusters):
        clusterinfo['cluster' + str(i)] = df[df['cluster_id'] == i].mean()
    clusterinfo = clusterinfo.drop('cluster_id')

    my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
    my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)


def decisoin_tree(df):
    # ライブラリのインポート
    import pandas as pd
    import sklearn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    # 必要なライブラリーをインポート
    import pandas as pd
    from sklearn import tree
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_diabetes
    import graphviz

    # 特徴量をダミー変数化
    dfcol = df.columns[17:]
    print(dfcol)
    df1 = df[dfcol]
    print(df1.head())
    # ターゲットの値を文字列から数値へ変換
    labelencoder = LabelEncoder()
    df['primaryCategory.categoryName'] = labelencoder.fit_transform(df['primaryCategory.categoryName'])
    # df1["primaryCategory"] = df['primaryCategory.categoryName']
    target = df['primaryCategory.categoryName']
    print(df1.head())
    print(df1.tail())

    # 決定木モデルの訓練
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(df1, target)
    # 訓練済みの決定木の視覚化
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=dfcol,
                                    class_names=dfcol,
                                    rounded=True,
                                    filled=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('data/binary_tree')


# df = pd.read_csv("data/ebay_categories_20190920.csv")
# decisoin_tree(df)