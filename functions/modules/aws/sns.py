import boto3

TOPIC_ARN = "arn:aws:sns:ap-northeast-1:291357645530:NotificationTopic"


class Sns:
    def __init__(self, topic_arn=TOPIC_ARN):
        # self.resource = boto3.resource("sns")
        self.client = boto3.client("sns")
        self.topic_arn = topic_arn

    def publish(self, message="", subject=""):
        # msg = urllib2.urlopen('https://classmethod.jp/recruit/').read()
        # subject = u'クラスメソッド 採用情報'
        # client = boto3.client('sns')

        request = {
            'TopicArn': self.topic_arn,
            'Message': message,
            'Subject': subject
        }
        response = self.client.publish(**request)

        return response
