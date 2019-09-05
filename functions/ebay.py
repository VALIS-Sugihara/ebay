import json
import requests
from ebaysdk.finding import Connection as finding
# from ebaysdk import nodeText


def test(event, context):
    f = finding()
    f.execute('findItemsAdvanced', {'keywords': 'shoes'})

    dom = f.response_dom()
    mydict = f.response_dict()
    myobj = f.response_obj()

    print(myobj.itemSearchURL)

    return

    # process the response via DOM
    items = dom.getElementsByTagName('item')

    for item in items:
        print(nodeText(item.getElementsByTagName('title')[0]))

    # r = requests.get('https://google.com')
    # print(r.status_code)
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


