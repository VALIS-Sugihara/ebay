import boto3


class Kinesis:
    region_name = "us-east-1"
    stream_name = "ebay-exporter-stream"

    def __init__(self):
        self.client = boto3.client('kinesis', region_name=self.region_name)

    def put_record(self, data):

        data = data

        response = self.client.put_record(
            StreamName='string',
            Data=b'bytes',
            PartitionKey='string',
            ExplicitHashKey='string',
            SequenceNumberForOrdering='string'
        )
        return response

    def put_records(self, data):
        data = data

        response = self.client.put_records(
            Records=[
                {
                    'Data': b'bytes',
                    'ExplicitHashKey': 'string',
                    'PartitionKey': 'string'
                },
            ],
            StreamName=self.stream_name
        )
        return response

    def get_records(self):
        response = self.client.get_records(
            ShardIterator='string',
            Limit=123
        )

    def list_shards(self):
        response = self.client.list_shards(
            StreamName=self.stream_name,
            # NextToken='string',
            # ExclusiveStartShardId='string',
            # MaxResults=123,
            # StreamCreationTimestamp=datetime(2015, 1, 1)
        )
        return response


import pandas as pd
df = pd.read_csv("../../data/ebay_categories_20191021.csv")
print(df.head())
exit()
k = Kinesis()
print(k.list_shards())