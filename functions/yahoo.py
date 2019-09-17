# -*- coding: utf-8 -*-

import json
import requests
import pandas as pd
import xmltodict

"""
Client ID：
    dj00aiZpPUlMRXp5U0loQmRXWiZzPWNvbnN1bWVyc2VjcmV0Jng9NTM-
シークレット：
    qU7QwCYwqbQShh7Z6166pLeW2uAmyIOIJ7XxvDLJ
"""
CLIENT_ID = "dj00aiZpPUlMRXp5U0loQmRXWiZzPWNvbnN1bWVyc2VjcmV0Jng9NTM-"
SECRET = "qU7QwCYwqbQShh7Z6166pLeW2uAmyIOIJ7XxvDLJ"


class Yahoo:
    """
    {
        'AuctionID': 'n357518070',
        'Title': '即決 美品 保証 ライカ Leica ライカTL2 シルバー [ボディ] 18188',
        'CategoryId': '2084305451',
        'Seller': {'Id': 'h54410', 'ItemListUrl': 'https://auctions.yahooapis.jp/AuctionWebService/V2/sellingList?sellerID=h54410', 'RatingUrl': 'https://auctions.yahooapis.jp/AuctionWebService/V1/ShowRating?id=h54410'},
        'ItemUrl': 'https://auctions.yahooapis.jp/AuctionWebService/V2/auctionItem?auctionID=n357518070',
        'AuctionItemUrl': 'https://page.auctions.yahoo.co.jp/jp/auction/n357518070',
        'Image': {'@width': '115', '@height': '100', '#text': 'https://wing-auctions.c.yimg.jp/sim?furl=auctions.c.yimg.jp/images.auctions.yahoo.co.jp/image/dr000/auc0507/users/0f2907125
761fd4ca5cfe7cc6bfe89d0bd91c1e4/i-img1200x1043-1563442277eo6exd993495.jpg&dc=1&sr.fs=20000'},
        'OriginalImageNum': '7',
        'CurrentPrice': '178000.00',
        'Bids': '0',
        'EndTime': '2019-09-09T19:43:41+09:00',
        'BidOrBuy': '178000.00',
        'IsReserved': 'false',
        'CharityOption': {'Proportion': '0'},
        'Option': {'FeaturedIcon': 'https://s.yimg.jp/images/auct/front/images/featured.gif', 'BuynowIcon': 'https://s.yimg.jp/images/auct/front/images/buynow.gif', 'EasyPaymentIcon': 'https://s.yimg.jp/images/pay/icon_s16.gif', 'IsBold': 'fal
se', 'IsBackGroundColor': 'false', 'IsOffer': 'false', 'IsCharity': 'false'},
        'IsAdult': 'false'
    }
    """
    column_permutations = ("Title", "CurrentPrice", "BidOrBuy", "shopName", "ItemUrl", "CategoryId",)
    property_permutations = ("Title", "CurrentPrice", "BidOrBuy", "shopName", "ItemUrl", "CategoryId",)

    def __init__(self):
        pass

    def search(self, query="", add_options={}):
        url = "https://auctions.yahooapis.jp/AuctionWebService/V2/search"
        item_parameters = {
            'appid': CLIENT_ID,
            'query': query,
            "page": 1,
            # "type": "all",  # all（全文一致） or any（部分一致）
            # "category": {id},
            # 'output': 'json',
        }
        if any(add_options) and isinstance(add_options, dict):
            item_parameters.update(add_options)
        response = requests.get(url, params=item_parameters)
        self._assert_response(response)
        return response

    def make_dataframe(self, items):
        df = pd.DataFrame(columns=list(self.column_permutations))
        for i, item in enumerate(items):
            df.loc[i] = self.get_values(item)

        return df

    def get_values(self, item):
        values = []

        def _get_value(key, item):
            if isinstance(key, str):
                return item.get(key, None)
            if isinstance(key, list):
                return " ".join([item.get(k) for k in key])
            if isinstance(key, dict):
                for k, v in key.items():
                    return _get_value(v, getattr(item, k))

        for key in self.property_permutations:
            values.append(_get_value(key, item))

        return values

    def _assert_response(self, response):
        # 接続確認
        assert response.status_code == 200, print("Response Error!! ", response.reply.ack)
        # 返却値（item）確認
        # assert type(response.reply.searchResult.item) == list, print("Invalid Item!! ", response.reply)

    def get_items(self, response):
        response = response.text

        # XML to json( dict )
        response = xmltodict.parse(response)
        response = json.dumps(response, indent=2)
        response = json.loads(response)

        return response["ResultSet"]["Result"]["Item"]

    def get_results(self, response):
        response = response.text

        # XML to json( dict )
        response = xmltodict.parse(response)
        response = json.dumps(response, indent=2)
        response = json.loads(response)

        return response["ResultSet"]["Result"]

    def get_total_pages(self, response):
        response = response.text

        # XML to json( dict )
        response = xmltodict.parse(response)
        response = json.dumps(response, indent=2)
        response = json.loads(response)

        return response["ResultSet"]["@totalResultsReturned"]

    def get_categories(self, category_id):
        url = "https://auctions.yahooapis.jp/AuctionWebService/V2/categoryTree"
        item_parameters = {
            'appid': CLIENT_ID,
            "category": category_id,
        }
        # if any(add_options) and isinstance(add_options, dict):
        #     item_parameters.update(add_options)
        response = requests.get(url, params=item_parameters)
        self._assert_response(response)
        return response



yahoo = Yahoo()
response = yahoo.get_categories("2084261684")
results = yahoo.get_results(response)
print(results)