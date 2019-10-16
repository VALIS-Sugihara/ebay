import re
import pandas as pd
import numpy as np
from functions.modules.google import Google

import spacy
import neologdn
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en')
nlp_ginza = spacy.load('ja_ginza_nopn')


def compare(en_list, ja_list):
    google = Google()

    columns = ["en", "ja2en", "ja", "en2ja", "score", "score_en", "score_ja"]
    df = pd.DataFrame(columns=columns)

    for en in en_list:
        doc_ebay_en = nlp(en)
        lst = []
        en2ja = google.translate(text=en, source="en", target="ja")
        doc_ebay_ja = nlp_ginza(en2ja)
        for ja in ja_list:
            # 特殊記号等の置換
            ja = neologdn.normalize(ja)
            ja = re.sub(u'[■-♯]', ' ', ja)

            doc_yahoo_ja = nlp_ginza(ja)
            ja2en = google.translate(text=ja, source="ja", target="en")
            doc_yahoo_en = nlp(ja2en)

            score_en = doc_ebay_en.similarity(doc_yahoo_en)
            score_ja = doc_ebay_ja.similarity(doc_yahoo_ja)

            # 平方根で平均化
            score = np.sqrt((score_en**2) * (score_ja**2) / 2)

            print(score_en, score_ja)

            lst.append([en, ja2en, ja, en2ja, score, score_en, score_ja])

        df = df.append(lst)

    df = df.sort_values(by=["score"], ascending=False)
    # df.to_csv("data/similarity.csv")

    return score


# ebay_df = pd.read_csv("data/sample_ebay_leica.csv")
# en_list = ebay_df["title"]

# from collections import Counter
# doc = nlp("".join(en_list))
# words = [token.text for token in doc]
# word_freq = Counter(words)
# common_words = word_freq.most_common(50)
# print(common_words)
# exit()

# yahoo_df = pd.read_csv("data/sample_yahoo_leica.csv")
# ja_list = yahoo_df["Title"]
# compare([en_list[0]], ja_list)
