import os
ENV = os.getenv("ENV")

import re
import pandas as pd
from ebaysdk.finding import Connection as finding
from modules.google import Google

# for get_model()
import urllib3
import certifi
from bs4 import BeautifulSoup

import spacy
# 英語のtokenizer、tagger、parser、NER、word vectorsをインポート
nlp = spacy.load('en_core_web_sm')


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
    config_file = "modules/ebay.yaml"
    _method_names = ("findCompletedItems", "findItemsAdvanced",)
    column_permutations = (
        "JP_shortTitle", "shortTitle",  "itemId", "title", "viewItemURL", "currentPrice", "currency", "country", "primaryCategory.categoryId",
        "primaryCategory.categoryName", "listingType", "condition"
    )
    property_permutations = (
        "JP_shortTitle", "shortTitle", "itemId", "title", "viewItemURL", {"sellingStatus": {"currentPrice": "value"}},
        {"sellingStatus": {"currentPrice": "_currencyId"}}, "country", {"primaryCategory": "categoryId"},
        {"primaryCategory": "categoryName"}, {"listingInfo": "listingType"}, {"condition": "conditionDisplayName"}
    )
    keywords = None

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
        self._set_keywords(keywords)
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
        google = Google()
        values = []

        def _get_short_title(item):
            ptn = r"[^a-zA-Z0-9\s]"
            title = re.sub(ptn, "", getattr(item, "title", ""))
            doc = nlp(title)
            # 名詞、固有名詞、数字以外を除去したカラムを作成
            words = []
            for token in doc:
                if token.pos_ in ("PROPN", "NOUN", "NUM",):
                    words.append(token.text)
            return " ".join(words)

            # lst = re.sub(ptn, "", getattr(item, "title", [])).lower().split()
            # # キーワードから３単語分抽出
            # try:
            #     target = int(lst.index(self._get_keywords()))
            # except:
            #     target = 0
            # length = target + 3
            # return " ".join(lst[target:length])

        def _get_value(key, item):
            if key == "shortTitle":
                return _get_short_title(item)
            if key == "JP_shortTitle":
                return None  # TEST
                return google.translate(text=_get_short_title(item), source="en", target="ja")
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
        return True
        # 接続確認
        assert response.reply.ack == 'Success', print("Response Error!! ", response.reply.ack)
        # 返却値（item）確認
        assert type(response.reply.searchResult.item) == list, print("Invalid Item!! ", response.reply)

    def get_items(self, response):
        try:
            items = response.reply.searchResult.item
        except AttributeError:
            items = [{}]
        return items

    def get_model(self, url):
        # for Series
        if not isinstance(url, str):
            url = url.to_list()
            for u in url:
                print(u)
                self.get_model(u)
        try:
            http = urllib3.PoolManager(
                cert_reqs='CERT_REQUIRED',
                ca_certs=certifi.where())
            r = http.request('GET', url)
            soup = BeautifulSoup(r.data, 'html.parser')
            t = soup.find_all("h2", attrs={"itemprop": "model"})
            print(t[0].text if any(t) and isinstance(t, list) else None)
            return t[0].text if any(t) and isinstance(t, list) else None
        except:
            print(None)
            return None

    def get_total_count(self, response):
        try:
            return response.reply.paginationOutput.totalEntries
        except:
            return 0

    def get_total_pages(self, response):
        try:
            return response.reply.paginationOutput.totalPages
        except:
            return 0

    def _set_keywords(self, keywords):
        self.keywords = keywords.split()[0]

    def _get_keywords(self):
        return self.keywords
