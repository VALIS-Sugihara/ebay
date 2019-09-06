import json
# import datetime
# import requests
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ebaysdk.finding import Connection as finding


def test(event, context):
    ebay = Ebay()

    page_number = 1
    add_options = {"paginationInput": {
        "entriesPerPage": 100,
        "pageNumber": page_number
    }}

    response = ebay.general_search(keywords="daiwa reel", add_options=add_options)
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

            response = ebay.general_search(keywords="daiwa reel", add_options=add_options)
            items = ebay.get_items(response)

            df = df.append(ebay.make_dataframe(items))

    # *** SearchDetails ***
    names = df.shortTitle
    counts = []
    for name in names:
        ptn = r"[<>]"
        name = re.sub(ptn, "", name)
        try:
            response = ebay.detail_search(keywords=name)
            cnt = ebay.get_total_count(response)
        except:
            cnt = None
        counts.append(cnt)

    df["TotalCounts"] = counts

    # JP産のもの
    df = df[df["country"] == "JP"]
    # 100USD 以上のもの
    # df = df[df["currentPrice"] >= 100]

    # CSV 出力
    df.to_csv("daiwa.csv")

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


class Ebay:
    """ response.reply.searchResult.item[0]
    {
        'itemId': '303272850097',
        'title': 'Vintage Daiwa 7000C Saltwater Spinning Fishing Reel for Repair or Parts - 7000 C',
        'globalId': 'EBAY-US',
        'primaryCategory': {
            'categoryId': '36147',
            'categoryName': 'Spinning Reels'
        },
        'galleryURL': 'https://thumbs2.ebaystatic.com/m/mNE4d2Tg4AxrKJ5JMtfcDYA/140.jpg',
        'viewItemURL': 'https://www.ebay.com/itm/Vintage-Daiwa-7000C-Saltwater-Spinning-Fishing-Reel-Repair-Parts-7000-C-/303272850097',
        'paymentMethod': 'PayPal',
        'autoPay': 'false',
        'postalCode': '33993',
        'location': 'Cape Coral,USA',
        'country': 'US',
        'shippingInfo': {
            'shippingType': 'Calculated',
            'shipToLocations': 'Worldwide',
            'expeditedShipping': 'true',
            'oneDayShippingAvailable': 'false',
            'handlingTime': '4'
        },
        'sellingStatus': {
            'currentPrice': {
                '_currencyId': 'USD',
                'value': '10.5'
            },
            'convertedCurrentPrice': {
                '_currencyId': 'USD',
                'value': '10.5'
            },
            'sellingState': 'Ended'
        },
        'listingInfo': {
            'bestOfferEnabled': 'false',
            'buyItNowAvailable': 'false',
            'startTime': datetime.datetime(2019, 9, 3, 3, 9),
            'endTime': datetime.datetime(2019, 9, 6, 0, 36, 18),
            'listingType': 'FixedPrice',
            'gift': 'false',
            'watchCount': '1'
        },
        'returnsAccepted': 'false',
        'condition': {
            'conditionId': '3000',
            'conditionDisplayName': 'Used'
        },
        'isMultiVariationListing': 'false',
        'topRatedListing': 'false'
    }
    """
    config_file = "ebay.yaml"
    _method_names = ("findCompletedItems", "findItemsAdvanced",)
    column_permutations = (
        "shortTitle", "itemId", "title", "currentPrice", "currency", "country", "primaryCategory.categoryId",
        "primaryCategory.categoryName", "listingType", "condition"
    )
    property_permutations = (
        "shortTitle", "itemId", "title", {"sellingStatus": {"currentPrice": "value"}},
        {"sellingStatus": {"currentPrice": "_currencyId"}}, "country", {"primaryCategory": "categoryId"},
        {"primaryCategory": "categoryName"}, {"listingInfo": "listingType"}, {"condition": "conditionDisplayName"}
    )

    def __init__(self):
        self.api = finding(config_file=self.config_file)

    def general_search(self, method_name="findCompletedItems", keywords=None, add_options={}):
        options = {
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
        options.update(add_options)
        return self.find_items(method_name=method_name, keywords=keywords, add_options=options)

    def detail_search(self, method_name="findCompletedItems", keywords=None, add_options={}):
        options = {
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
        options.update(add_options)
        return self.find_items(method_name=method_name, keywords=keywords, add_options=options)

    def find_items(self, method_name="findItemsAdvanced", keywords=None, add_options={}):
        options = {
            # キーワード
            "keywords": keywords,
        }
        # TODO:: update
        if any(add_options) and isinstance(add_options, dict):
            options.update(add_options)

        response = self.api.execute(method_name, options)

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
            if key == "shortTitle":
                return " ".join(getattr(item, "title", []).split()[:4])
            if isinstance(key, str):
                return getattr(item, key, None)
            if isinstance(key, list):
                return " ".join([getattr(item, k) for k in key])
            if isinstance(key, dict):
                for k, v in key.items():
                    return _get_value(v, getattr(item, k))

        for key in self.property_permutations:
            values.append(_get_value(key, item))

        return values

    def _assert_response(self, response):
        # 接続確認
        assert response.reply.ack == 'Success', print("Response Error!! ", response.reply.ack)
        # 返却値（item）確認
        assert type(response.reply.searchResult.item) == list, print("Invalid Item!! ", response.reply)

    def get_items(self, response):
        return response.reply.searchResult.item

    def get_total_count(self, response):
        return response.reply.paginationOutput.totalEntries

    def get_total_pages(self, response):
        return response.reply.paginationOutput.totalPages
