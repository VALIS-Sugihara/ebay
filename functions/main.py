import json
import re
import pandas as pd
from ebay import Ebay
from rakuten import Rakuten
from yahoo import Yahoo


def ebay(event, context):
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

    # CSV 出力
    df.to_csv("./data/sample_ebay_detail_%s.csv" % (keywords,))

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
    #
    # df["TotalCounts"] = counts

    # JP産のもの
    df = df[df["country"] == "JP"]
    # 100USD 以上のもの
    # df = df[df["currentPrice"] >= 100]
    # count数でソート
    # df = df.sort_values(by=["TotalCounts"], ascending=False)

    # CSV 出力
    df.to_csv("./data/sample_ebay_detail_%s.csv" % (keywords,))

    # モデル取得
    df = df.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))

    # CSV 出力
    df.to_csv("./data/sample_ebay_detail_%s_model.csv" % (keywords,))

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def rakuten(event, context):
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
    df.to_csv("./data/sample_rakuten_%s.csv" % (keyword,))

    # モデル取得
    # df = df.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))

    # CSV 出力
    # df.to_csv("./data/sample_ebay_detail_%s_model.csv") % (keywords,)

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


def yahoo(event, context):
    yahoo = Yahoo()

    """
        STEP 1 : キーワード Ebay 検索
    """
    page_number = 1
    add_options = {
        "page": page_number
    }

    # keywords = "daiwa"
    # query = "ニコン"
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
    # df.to_csv("./data/sample_rakuten_%s.csv") % (keyword,)

    # モデル取得
    # df = df.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))

    # CSV 出力
    df.to_csv("./data/sample_yahoo_%s.csv" % (query,))

    return df

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


# ebay(True, True)

ebay = Ebay()
yahoo = Yahoo()
df_ebay = pd.read_csv("data/sample_ebay_detail_nikon.csv")
df_ebay = pd.read_csv("data/sample_ebay_detail_nikon.csv")
import similarity

columns = ["en", "ja2en", "ja", "en2ja", "score", "score_en", "score_ja"]
df_similarity = pd.DataFrame(columns=columns)

for i in range(0, len(df_ebay)):
    colum = df_ebay.loc[i]
    print(colum["title"])
    if colum["model"] is not None:
        query = "Nikon %s" % (colum["model"])
        event = {"query": query}
        df_yahoo = yahoo(event, True)
        print(df_yahoo.head())

        en_list = [colum["title"]]
        ja_list = df_yahoo["Title"]
        df = similarity.compare(en_list, ja_list)
        df_similarity.append(df.loc[0])
        print(df_similarity.head())
        exit()
df_ebay = df_ebay.assign(model=lambda x: ebay.get_model(x["viewItemURL"]))
print(df_ebay.head())
df_ebay.to_csv("data/sample_ebay_detail_nikon_model.csv")



# yahoo = Yahoo()
# query =
# yahoo.search(query="Nikon")