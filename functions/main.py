import json
import re
import datetime
import pandas as pd
import numpy as np
from ebay import Ebay
from rakuten import Rakuten
from yahoo import Yahoo
from google import Google
import spacy
from sklearn.preprocessing import LabelEncoder

# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_md')
nlp_ginza = spacy.load('ja_ginza_nopn')

TODAY = datetime.datetime.now().date().strftime("%Y%m%d")


def ebay2df(event, context):
    ebay = Ebay()

    """
    STEP 1 : キーワード Ebay 検索
    """
    page_number = 1
    add_options = {"paginationInput": {
        "entriesPerPage": 100,
        "pageNumber": page_number
    }}

    keywords = event["keywords"]

    response = ebay.general_search(keywords=keywords, add_options=add_options)
    items = ebay.get_items(response)
    if not any(items):
        return False

    pages = int(ebay.get_total_pages(response))

    df = ebay.make_dataframe(items)

    print(df.head())

    # Get All Pages.
    if pages > 1:
        for i in range(1, pages):
            # if i < 100:
            if i < 50:  # TEST:: < 50p
                add_options = {"paginationInput": {
                    "entriesPerPage": 100,
                    "pageNumber": i + 1
                }}

                response = ebay.general_search(keywords=keywords, add_options=add_options)
                items = ebay.get_items(response)

                if not any(items):
                    break

                df = df.append(ebay.make_dataframe(items))

                print(df.tail())

    # JP産のもの
    df = df[df["country"] == "JP"]
    # 100USD 以上のもの
    # df = df[df["currentPrice"] >= 100]

    # CSV 出力
    df.to_csv("./data/ebay_detail_%s_%s.csv" % (keywords, TODAY,))

    # モデル取得
    # df = df.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))
    view_item_urls = df["viewItemURL"]
    url_list = []
    for url in view_item_urls:
        url_list.append(ebay.get_model(url))
    df["model"] = url_list

    print(df.head())

    """
        STEP 2 : キーワード Ebay 検索
    """
    s_titles = df.shortTitle
    models = df.model
    category_ids = df["primaryCategory.categoryId"]

    counts = []
    for title, model, category_id in zip(s_titles, models, category_ids):
        try:
            name = model if model is not None else title
            add_options = {"categoryId": category_id}
            response = ebay.detail_search(keywords=name, add_options=add_options)
            cnt = int(ebay.get_total_count(response))
            print(name, cnt)
        except:
            cnt = None
        counts.append(cnt)

    df["TotalCounts"] = counts

    # count数でソート
    df = df.sort_values(by=["TotalCounts"], ascending=False)

    # ターゲットの値を文字列から数値へ変換
    labelencoder = LabelEncoder()
    df['Target'] = labelencoder.fit_transform(df['primaryCategory.categoryName'])

    # CSV 出力
    df.to_csv("./data/ebay_detail_%s_model_%s.csv" % (keywords, TODAY,))

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def ebay_market_price(df):
    keywords = "nikon"
    df = df.groupby(["model", "primaryCategory.categoryId"], as_index=False).mean()
    df.to_csv("data/ebay_market_price_%s.csv" % (keywords,))


def rakuten2df(event, context):
    rakuten = Rakuten()

    """
        STEP 1 : キーワード Ebay 検索
    """
    page_number = 1
    add_options = {
        "page": page_number
    }

    # keywords = "daiwa"
    keyword = "ニコン"

    response = rakuten.search(keyword=keyword, add_options=add_options)
    pages = int(rakuten.get_total_pages(response))
    items = rakuten.get_items(response)
    print(items[0])

    df = rakuten.make_dataframe(items)

    # Get All Pages.
    if pages > 1:
        for i in range(1, pages):
            add_options = {
                "page": i + 1
            }

            response = rakuten.search(keyword=keyword, add_options=add_options)
            items = rakuten.get_items(response)

            df = df.append(rakuten.make_dataframe(items))

    """
        STEP 2 : キーワード Ebay 検索
    """
    # names = df.shortTitle
    # counts = []
    # for name in names:
    #     try:
    #         response = ebay.detail_search(keywords=name)
    #         cnt = ebay.get_total_count(response)
    #     except:
    #         cnt = None
    #     print(name, cnt)
    #     counts.append(cnt)
        # TEST
        # counts.append(None)

    # df["TotalCounts"] = counts

    # JP産のもの
    # df = df[df["country"] == "JP"]
    # 100USD 以上のもの
    # df = df[df["currentPrice"] >= 100]
    # count数でソート
    # df = df.sort_values(by=["TotalCounts"], ascending=False)

    # CSV 出力
    # df.to_csv("./data/sample_rakuten_%s.csv" % (keyword,), encoding="s-jis")
    df.to_csv("./data/sample_rakuten_%s_%s.csv" % (keyword, TODAY,))

    # モデル取得
    # df = df.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))

    # CSV 出力
    # df.to_csv("./data/sample_ebay_detail_%s_model.csv") % (keywords,)

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def yahoo2df(event, context):
    yahoo = Yahoo()

    """
        STEP 1 : キーワード Ebay 検索
    """
    page_number = 1
    add_options = {
        "page": page_number
    }

    query = event["query"]

    response = yahoo.search(query=query, add_options=add_options)

    pages = int(yahoo.get_total_pages(response))

    items = yahoo.get_items(response)

    df = yahoo.make_dataframe(items)

    # Get All Pages.
    if pages > 1:
        # if pages > 50:
        #     pages = 50
        for i in range(1, pages):
            add_options = {
                "page": i + 1
            }

            response = yahoo.search(query=query, add_options=add_options)
            items = yahoo.get_items(response)

            df = df.append(yahoo.make_dataframe(items))

    """
        STEP 2 : 英翻訳、shortTitlte
    """
    def get_en_title(title):
        google = Google()
        return google.translate(text=title, source="ja", target="en")

    df["en_Title"] = df.Title.apply(get_en_title)

    def get_en_short_title(en_title):
        ptn = r"[^a-zA-Z0-9\s]"
        title = re.sub(ptn, "", en_title)
        doc = nlp(title)
        # 名詞、固有名詞、数字以外を除去したカラムを作成
        words = []
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN", "NUM",):
                words.append(token.text)
        return " ".join(words)

    df["en_short_Title"] = df.en_Title.apply(get_en_short_title)

    def get_ja_short_title(title):
        # ptn = r"[^a-zA-Z0-9\s]"
        # title = re.sub(ptn, "", title)
        doc = nlp_ginza(title)
        # 名詞、固有名詞、数字以外を除去したカラムを作成
        words = []
        for token in doc:
            print(token.pos_)
            if token.pos_ in ("PROPN", "NOUN", "NUM",):
                words.append(token.text)
        return " ".join(words)


    df["short_Title"] = df.Title.apply(get_ja_short_title)

    # CSV 出力
    df.to_csv("./data/yahoo_%s_%s.csv" % (query, TODAY,))

    return df

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def similarity(ebay_df, yahoo_df):
    """ 相関性の高いレコードを連結 """
    shortTitles = ebay_df["shortTitle"].to_list()
    models = ebay_df["model"].to_list()
    en_short_Titles = yahoo_df["en_short_Title"].to_list()
    targets = ebay_df["Target"].to_list()
    df = pd.DataFrame(columns=yahoo_df.columns)

    def same_category(i, arr, targets, yahoo_df):
        max_index = arr.index(max(arr))
        if str(yahoo_df.at[max_index, "Target_y"]) == str(targets[i]):
            return max_index
        else:
            arr.pop(max_index)
            same_category(i, arr, targets, yahoo_df)

    for i, sttl in enumerate(shortTitles):
        print("sttl is ...", sttl)
        try:
            sttl_doc = nlp(str(sttl))
            model_doc = nlp(str(models[i])) if models[i] is not None else nlp("")
            # shortTitle: en_short_Title, model: en_short_Title の相関平均で比較する
            arr = []
            for ensttl in en_short_Titles:
                print("ensttl is ...", ensttl)
                ensttl_doc = nlp(str(ensttl))
                score1 = sttl_doc.similarity(ensttl_doc)
                score2 = model_doc.similarity(ensttl_doc)
                arr.append(np.mean([score1, score2]))
                print("score is ...", np.mean([score1, score2]))
                # print(arr)
            max_index = same_category(i, arr, targets, yahoo_df)
            print(max_index)
            df = df.append(yahoo_df.loc[max_index:max_index])
            print(df.tail())
        except ValueError as e:
            print(e)
            d = {}
            for c in df.columns:
                d[c] = None
            empty_list = pd.DataFrame(d, columns=df.columns)
            df = df.append(empty_list, ignore_index=True)
            print(df.tail())
            continue

    df = df.reset_index(drop=True)
    print(df.head())
    print(df.shape, ebay_df.shape)
    df3 = pd.concat([ebay_df, df], axis=1)
    print(df3.head())
    df3.to_csv("data/ebay_yahoo_detail_%s.csv" % (TODAY,))


def plot(df):
    import seaborn as sns
    # 特徴量の散布図行列
    sns.pairplot(data=df, hue='type')


# from machine_learnings import *
import machine_learnings

def exec_all(keywords="nikon"):
    import os
    try:
        # ebay2df({"keywords": keywords}, True)
        # yahoo2df({"query": keywords}, True)

        machine_learnings.categories(keywords)
        machine_learnings.ml(keywords)

        ebay_df = pd.read_csv("data/ebay_detail_%s_model_%s.csv" % (keywords, TODAY,))
        yahoo_df = pd.read_csv("data/yahoo_%s_%s.csv" % (keywords, TODAY,))
        similarity(ebay_df, yahoo_df)
        os.system("aws s3 cp _result s3://ebay-frontend/data/%s_success" % (TODAY,))
    except Exception:
        import traceback
        traceback.print_exc()
        os.system("aws s3 cp _result s3://ebay-frontend/data/%s_error" % (TODAY,))


exec_all("nikon lens mf")

# ebay2df(True, True)
# yahoo2df({"query": "nikon"}, True)
# TODAY = "20191007"
# ebay_df = pd.read_csv("data/ebay_detail_nikon_model_%s.csv" % (TODAY,))
# yahoo_df = pd.read_csv("data/yahoo_nikon_%s.csv" % (TODAY,))
# similarity(ebay_df, yahoo_df)

# csv = "data/ebay_categories_20190920.csv"
# df = pd.read_csv(csv)
# plot(df)
