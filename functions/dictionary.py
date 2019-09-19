import json
import re
from google import Google
google = Google()

import redis
# Redis に接続します
r = redis.Redis(host='localhost', port=6379, db=0)


class MultiDimensionalArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                # for bytes
                bytes_flg = False
                list_item = []
                for i in item:
                    if isinstance(i, bytes):
                        list_item.append(i.decode())
                        bytes_flg = True
                    else:
                        list_item.append(i)
                if bytes_flg is True:
                    item = tuple(list_item)
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            else:
                return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


enc = MultiDimensionalArrayEncoder()
jsonstring = enc.encode([1, 2, (3, 4), [5, 6, (7, 8)]])

# print(jsonstring)

# [1, 2, {"items": [3, 4], "__tuple__": true}, [5, 6, {"items": [7, 8], "__tuple__": true}]]

# print(json.loads(jsonstring, object_hook=hinted_tuple_hook))

# [1, 2, (3, 4), [5, 6, (7, 8)]]


class Dictionary:
    """
    各翻訳単語やブランド毎の頻出単語を操作するクラス(in Redis)
    """
    def __init__(self, lang="ja", target="en"):
        self.lang = lang
        self.target = target
        self.enc = MultiDimensionalArrayEncoder()

    def to_json(self, origin):
        return self.enc.encode(origin)

    def from_json(self, jsonstring):
        return json.loads(jsonstring, object_hook=hinted_tuple_hook)

    def get(self, word=""):
        result = r.get(word)
        if isinstance(result, bytes):
            result.decode()
        return result

    def set(self, key, value):
        r.set(key, value)

    def register(self, word=""):
        text = self.get(word)
        if text is None:
            # en に翻訳
            ptn = r"[0-9]+"
            if not re.match(ptn, word):
                value = google.translate(text=word, source=self.lang, target=self.target)
            else:
                value = word
            self.set(word, value)
            return value
        else:
            return text


class FrequentDictionary(Dictionary):
    key = "common_words"

    def __init__(self, brand="ebay"):
        super(FrequentDictionary, self).__init__()
        self.brand = brand

    def get_frequency(self, brand="ebay"):
        return r.smembers(brand+":"+self.key)

    def set_frequency(self, brand, key, value):
        value = self.to_json(value)
        r.sadd(brand+":"+self.key+":"+key, value)


# print(r.keys("ebay*"))
# print(r.smembers("ebay:common_words:Film Cameras"))