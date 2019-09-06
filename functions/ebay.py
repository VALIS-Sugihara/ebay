import json
# import datetime
# import requests
from ebaysdk.finding import Connection as finding


def test(event, context):
    ebay = Ebay()
    # response = ebay.general_search(keywords="daiwa reel")
    # item = ebay._get_items(response)[0]
    # print(item)

    # title = item.title
    # detail_keywords = " ".join(title.split(" ")[0:3])
    # print("detail_keywords: ", detail_keywords)
    # response = ebay.detail_search(keywords=detail_keywords)
    response = ebay.detail_search(keywords="daiwa reel")
    item = ebay._get_items(response)[1]
    print(item)

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


class Ebay:

    config_file = "ebay.yaml"
    _method_names = ("findCompletedItems", "findItemsAdvanced",)

    def __init__(self):
        self.api = finding(config_file=self.config_file)

    def general_search(self, method_name="findItemsAdvanced", keywords=None, add_options={}):
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
                # Sold Items
                {
                    "name": "SoldItemsOnly",
                    "value": "true"
                }
            ]
        }
        return self.find_items(method_name=method_name, keywords=keywords, add_options=add_options)

    def detail_search(self, method_name="findCompletedItems", keywords=None, add_options={}):
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
                # Sold Items
                {
                    "name": "SoldItemsOnly",
                    "value": "true"
                }
            ]
        }
        return self.find_items(method_name=method_name, keywords=keywords, add_options=add_options)

    def find_items(self, method_name="findItemsAdvanced", keywords=None, add_options={}):
        options = {
            # キーワード
            "keywords": keywords,
            # ページネーション
            "paginationInput": {
                "entriesPerPage": 100,
                "pageNumber": 1
            },
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
        # TODO:: update
        if any(options) and isinstance(add_options, dict):
            options.update(add_options)

        response = self.api.execute(method_name, options)

        self._assert_response(response)

        return response

    def _assert_response(self, response):
        print(response.reply)
        # 接続確認
        assert response.reply.ack == 'Success', print("Response Error!! ", response.reply.ack)
        # 返却値（item）確認
        assert type(response.reply.searchResult.item) == list, print("Invalid Item!! ", response.reply)

    def _get_items(self, response):
        return response.reply.searchResult.item
