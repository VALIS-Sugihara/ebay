import json
import re
import datetime
import pandas as pd
from ebay import Ebay
from rakuten import Rakuten
from yahoo import Yahoo
from google import Google
import spacy
# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_sm')
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

    keywords = "nikon"

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
            cnt = ebay.get_total_count(response)
            print(name, cnt)
        except:
            cnt = None
        counts.append(cnt)

    df["TotalCounts"] = counts

    # count数でソート
    df = df.sort_values(by=["TotalCounts"], ascending=False)

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


# ebay2df(True, True)
# ebay(True, True)
# df = pd.read_csv("data/sample_ebay_detail_nikon_model.csv")
# ebay_market_price(df)

# yahoo2df({"query": "nikon"}, True)

def simlarity(ebay_df, yahoo_df):
    """ 相関性の高いレコードを連結 """
    ebay_df = pd.read_csv("data/ebay_detail_nikon_model_20190920.csv")
    yahoo_df = pd.read_csv("data/yahoo_nikon_20190920_ja.csv")

    shortTitles = ebay_df["shortTitle"]
    en_short_Titles = yahoo_df["en_short_Title"]
    df = pd.DataFrame(columns=yahoo_df.columns)
    for sttl in shortTitles:
        sttl_doc = nlp(sttl)
        arr = [sttl_doc.similarity(nlp(ensttl)) for ensttl in en_short_Titles]
        max_index = arr.index(max(arr))
        print(max_index)
        df = df.append(yahoo_df.loc[max_index:max_index])

    df = df.reset_index(drop=True)
    print(df.head())
    print(df.shape, ebay_df.shape)
    df3 = pd.concat([ebay_df, df], axis=1)
    print(df3.head())
    df3.to_csv("data/ebay_yahoo_detail_%s.csv" % (TODAY,))


df = pd.read_csv("data/ebay_yahoo_detail_20190920.csv")

def change_url(text):
    ptn = r".+\?auctionID=(.+)"
    id_ = re.match(ptn, text)
    base_url = "https://page.auctions.yahoo.co.jp/jp/auction/"
    url = base_url + id_.groups()[0] if id_ is not None else base_url
    return url

df["ItemUrl"] = df.ItemUrl.apply(change_url)
df.to_csv("data/ebay_yahoo_detail_20190920_b.csv")
# df2["en_short_Title"]

# print(df1.head())
# print(df2.head())


# df3 = pd.concat([df1, df2], axis=1)


# from google import Google
# google = Google()
# df = df.assign(en_Title=df.apply(lambda x: google.translate(text=x["Title"], source="ja", target="en"), axis=1))
# df.to_csv("data/svn_yahoo_nikon_en.csv")
# print(df.head())

# ebay = Ebay()
# yahoo = Yahoo()
# df_ebay = pd.read_csv("data/sample_ebay_detail_nikon.csv")
# df_ebay = pd.read_csv("data/sample_ebay_detail_nikon.csv")
# import similarity
#
# columns = ["en", "ja2en", "ja", "en2ja", "score", "score_en", "score_ja"]
# df_similarity = pd.DataFrame(columns=columns)
#
# for i in range(0, len(df_ebay)):
#     colum = df_ebay.loc[i]
#     print(colum["title"])
#     if colum["model"] is not None:
#         query = "Nikon %s" % (colum["model"])
#         event = {"query": query}
#         df_yahoo = yahoo(event, True)
#         print(df_yahoo.head())
#
#         en_list = [colum["title"]]
#         ja_list = df_yahoo["Title"]
#         df = similarity.compare(en_list, ja_list)
#         df_similarity.append(df.loc[0])
#         print(df_similarity.head())
#         exit()
# df_ebay = df_ebay.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))
# print(df_ebay.head())
# df_ebay.to_csv("data/sample_ebay_detail_nikon_model.csv")
