import json
import re
from ebay import Ebay
from rakuten import Rakuten


def test(event, context):
    ebay = Ebay()
    rakuten = Rakuten()

    """
        STEP 1 : キーワード Ebay 検索
    """
    page_number = 1
    add_options = {"paginationInput": {
        "entriesPerPage": 100,
        "pageNumber": page_number
    }}

    keywords = "leica"

    response = ebay.general_search(keywords=keywords, add_options=add_options)
    items = ebay.get_items(response)

    pages = int(ebay.get_total_pages(response))

    df = ebay.make_dataframe(items)

    # Get All Pages.
    if pages > 1:
        for i in range(1, pages):
            add_options = {"paginationInput": {
                "entriesPerPage": 100,
                "pageNumber": i + 1
            }}

            response = ebay.general_search(keywords=keywords, add_options=add_options)
            items = ebay.get_items(response)

            df = df.append(ebay.make_dataframe(items))

    """
        STEP 2 : キーワード Ebay 検索
    """
    names = df.shortTitle
    counts = []
    for name in names:
        # ptn = r"[^a-zA-Z0-9]"
        # name = re.sub(ptn, "", name)
        try:
            response = ebay.detail_search(keywords=name)
            cnt = ebay.get_total_count(response)
        except:
            cnt = None
        # print(name, cnt)
        counts.append(cnt)
        # TEST
        # counts.append(None)

    df["TotalCounts"] = counts

    # JP産のもの
    df = df[df["country"] == "JP"]
    # 100USD 以上のもの
    # df = df[df["currentPrice"] >= 100]
    # count数でソート
    df = df.sort_values(by=["TotalCounts"], ascending=False)

    # CSV 出力
    df.to_csv("./data/sample_ebay_detail_leica.csv")

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


test(True, True)