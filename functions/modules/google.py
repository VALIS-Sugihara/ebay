import requests


class Google:
    # GAS webapp in Translation Project
    url = "https://script.google.com/macros/s/AKfycbytv3cdchwPyuVI6f9wLtpvSqT_pyfMcEMMhNI9xeGPAWxxtqXu/exec"

    def __init__(self):
        pass

    def translate(self, text="", source="en", target="ja"):
        query = "?text=%s&source=%s&target=%s" % (text, source, target,)
        url = self.url + query
        response = requests.get(url)

        if response.status_code != 200:
            raise ConnectionError

        return response.text
