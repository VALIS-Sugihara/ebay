# -*- coding: utf-8 -*-

import requests
import pandas as pd
"""
アプリID/デベロッパーID
( / developerId)
    1096454421396465244
application_secret
    8cf4199913281446e362ac96affe4fdd7e8c7b5a
アフィリエイトID
(affiliateId)
    13f8577c.fe41f1cb.13f8577d.55f2e982
コールバック許可ドメイン
    valis.jp
ポリシーURL

"""
APPLICATION_ID = "1096454421396465244"
APPLICATION_SECRET = "8cf4199913281446e362ac96affe4fdd7e8c7b5a"
AFFILIATE_ID = "13f8577c.fe41f1cb.13f8577d.55f2e982"
CALLBACK_DOMAIN = "valis.jp"


class Rakuten:
    """
    {
        'mediumImageUrls': ['https://thumbnail.image.rakuten.co.jp/@0_mall/mitsuba/cabinet/leica/imgrc0080618667.jpg?_ex=128x128', 'https://thumbnail.image.rakuten.co.jp/@0_mall/mit
suba/cabinet/leica/imgrc0080618668.jpg?_ex=128x128', 'https://thumbnail.image.rakuten.co.jp/@0_mall/mitsuba/cabinet/leica/imgrc0080618669.jpg?_ex=128x128'],
        'pointRate': 2,
        'shopOfTheYearFlag': 0,
        'affiliateRate': 2,
        'shipOverseasFlag': 0,
        'asurakuFlag': 0,
        'endTime': '',
        'taxFlag': 0,
        'startTime': '',
        'itemCaption': '【※受注後発注/取り寄せ品のためキャンセル不可商品となります。】 ※この商品は多くのご予約を頂いておりますが、ドイツ本国よりの入荷が少なく、納入にお時間を頂いております。 ※10月1日に予定されております消費税増税以降に納入がずれ込む可能性もございます。その際は消費税は増税後の割合になります。消費税増税分の値引きは法律違反になるためお受けできません。 ※当店は『キャッシュレス・消費者還元事業者』にあたりますので、キャッシュレス決済でお申し込みいただければ各キャッシュレス業者が定める方法で還元がある予定です。 ・有効4730万画素フルサイズセンサーを採用。細部まで鮮明な静止画や4K動画が撮影できるほか、最高ISO感度50000までの高感度撮影も可能。 ・ほこりや水滴の侵入を防ぐ特殊な保護シーリング装備。 ・「ライカQ2」は、解放F値がF1.7と明るいズミルックス28mm単焦点レンズを搭載し、背景を美しくぼかして被写体が鮮明に際立つような表現も可能。 ・優れた描写性能により、光量が少ないシーンでも美しく撮影可能。 ・高精細368万ドットの有機EL電子ビューファインダーを搭載。高精細・高コントラストで深みのある色を実現。 ・オートフォーカスは0.15秒未満でのシャープなピント合わせを実現。 ・4K動画（3840×2160、30fps/24fps）撮影機能を搭載し、迫力あるリアルな映像表現が可能。 コード\u3000：19050 JAN:4548182190509 型式：35 mmフルサイズセンサー搭載コンパクトデジタルカメラ 撮像
フォーマット/アスペクト比：36×24mm/3:2 レンズ構成：9群11枚\u3000非球面レンズ：3枚 デジタルフレーム：28mm、35mm、50mm、75mmの各レンズ相当の画角 手ブレ補正機能：光学式（静止画および動画） 絞り\u3000\u3000：F1.7〜F16（1/3EVステップ） 撮像素子/画素数：フルサイズCMOSセンサー \u3000総画素数：5040万画素\u3000有効画素数：4730万画素 記録媒体：SDメモリーカード、SDHCメモリーカード、SDXCメモリーカード（UHS-II対応を推奨） 本体 ：丈夫で軽量なマグネシウム合金製 フィルターサイズ:E49 49mm 寸法（幅×高さ×奥行） 約130×80×91.9mm 質量 約718g（バッテリー含む）/約637g（バッテリー含まず） 付属品 キャリングストラップ、レンズフード、レンズキャップ、ホットシューカバー、充電式リチウムイオンバッテリー、バッテリーチャージャー、電源コード 対応アプリ Leica FOTOS App 【保証書について】 『LEICA Q2』には保証書は付属しておりません。保証期間中（ご購入後2年間）に修理等が必要になった場合には
、ご購入時の『お買い上げ明細書』と『3年保険通知書』のご提示をお願いします。修理は当店かライカジャパンにてお預かりいたします。 【※受注後発注/取り寄せ品のためキャンセル不可商品となります。】',
        'catchcopy': '【当店限定！ポイント2倍!!】[3年保険付]',
        'tagIds': [1000886, 1004718, 1005087, 1005094, 1005105],
        'smallImageUrls': ['https://thumbnail.image.rakuten.co.jp/@0_mall/mitsuba/cabinet/leica/imgrc0080618667.jpg?_ex=64x64', 'https://thumbnail.image.rakuten.co.jp/@0_mall/mitsuba/cabinet/leica/imgrc0080618668.jpg?_ex=64x64', 'https://thumbnail.image.rakuten.co.jp/@0_mall/mitsuba/cabinet/leica/imgrc0080618669.jpg?_ex=64x64'],
        'asurakuClosingTime': '',
        'imageFlag': 1,
        'availability': 1,
        'shopAffiliateUrl': '',
        'itemCode': 'mitsuba:10015896',
        'postageFlag': 0,
        'itemName': '[3年保険付] Leica Q2 #19050 フルサイズセンサー & SUMMILUX 28mm ハイエンドコンパクトデジカメ『納期4ヶ月程度』[※受注後発注/取り寄せ品のためキャンセル不可商品][02P05Nov16]',
        'itemPrice': 666900,
        'pointRateEndTime': '2019-09-12 12:59',
        'shopCode': 'mitsuba',
        'affiliateUrl': '',
        'giftFlag': 0,
        'shopName': 'カメラのミツバ',
        'reviewCount': 1,
        'asurakuArea': '',
        'shopUrl': 'https://www.rakuten.co.jp/mitsuba/',
        'creditCardFlag': 1,
        'reviewAverage': 1,
        'shipOverseasArea': '',
        'genreId': '110110',
        'pointRateStartTime': '2019-09-05 22:00',
        'itemUrl': 'https://item.rakuten.co.jp/mitsuba/leica-q2/'
    }
    """
    column_permutations = ("itemName", "itemPrice", "shopName", "itemUrl", "genreId")
    property_permutations = ("itemName", "itemPrice", "shopName", "itemUrl", "genreId")

    def __init__(self):
        pass

    def search(self, keyword=None, add_options={}):
        url = "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20170706"
        # item_url = 'https://app.rakuten.co.jp/services/api/IchibaGenre/Search/20140222'
        item_parameters = {
            'applicationId': APPLICATION_ID,
            'format': 'json',
            'formatVersion': 2,
            # 'genreId': 0,
            "keyword": keyword,
            "page": 1
        }
        if any(add_options) and isinstance(add_options, dict):
            item_parameters.update(add_options)
        response = requests.get(url, params=item_parameters)
        response = response.json()
        return response

    def get_items(self, response):
        items = response["Items"] if "Items" in response else [{}]
        return items

    def make_dataframe(self, items):
        df = pd.DataFrame(columns=list(self.column_permutations))
        for i, item in enumerate(items):
            df.loc[i] = self.get_values(item)

        return df

    def get_values(self, item):
        values = []

        def _get_value(key, item):
            if isinstance(key, str):
                return item.get(key, "")
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

    def get_total_count(self, response):
        return response["count"]

    def get_total_pages(self, response):
        return response["pageCount"]
