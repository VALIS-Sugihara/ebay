import pandas as pd
from modules.google import Google
from modules.yahoo import Yahoo
from modules.ebay import Ebay
from modules.decorators import print_func
import re
import spacy
import datetime
# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_md')
nlp_ginza = spacy.load('ja_ginza_nopn')

TODAY = datetime.datetime.now().date().strftime("%Y%m%d")


@print_func
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
    # TEST
    # query = "nikkor"

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
        # return " ".join(words)
        ptn = r"[^a-zA-Z0-9\s]"
        return re.sub(ptn, "", " ".join(words))

    df["short_Title"] = df.Title.apply(get_ja_short_title)

    # TEST
    # query = event["query"]

    # CSV 出力
    df.to_csv("./dummy/yahoo_%s_%s.csv" % (query, TODAY,))

    return df


def hot_selling(keywords):
    from modelnumbers.sekonic import Sekonic
    from modelnumbers.pre_limit import PreLimit

    ebay = Ebay()

    # item_list = Sekonic.modelnumbers
    item_list = PreLimit.modelnumbers
    # item_list = pd.read_csv("modelnumbers/pentax.csv").model.to_list()

    result = []
    for item in item_list:
        if item.strip() == "":
            continue
        item = str(item)
        # NOW
        add_options = {
            "itemFilter": [
                # Used
                {
                    "name": "Condition",
                    "value": 3000
                },
                # not Auction
                {
                    "name": "ListingType",
                    "value": "FixedPrice"
                },
            ]
        }
        response = ebay.general_search(method_name="findItemsAdvanced", keywords=item, add_options=add_options)
        now_cnt = ebay.get_total_count(response)
        # Sold
        response = ebay.general_search(method_name="findCompletedItems", keywords=item, add_options={})
        sold_cnt = ebay.get_total_count(response)
        print("======", item, "======")
        print("now_cnt: %s | sold_cnt: %s" % (now_cnt, sold_cnt,))

        if 0 < int(now_cnt) < int(sold_cnt):
            result.append(item)

    print("result is ...", result)
    return result


result = hot_selling("sekonic")
# for item in result:
#     yahoo2df({"query": item}, True)

# df = pd.read_csv("data/ebay_detail_pentax_model_20191031.csv")
# df1 = pd.DataFrame()
# df1["model"] = df.model.unique()
# print(df1.head())
# df1.to_csv("modelnumbers/pentax.csv")
