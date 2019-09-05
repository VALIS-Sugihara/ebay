import json
import requests


def test(event, context):
    # r = requests.get('https://google.com')
    # print(r.status_code)
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
