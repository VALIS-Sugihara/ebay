import boto3
import numpy as np

class Dynamodb():
    region_name = "us-east-1"

    def __init__(self):
        self.client = boto3.client('dynamodb', region_name=self.region_name)

    def put_item(self, table_name, item):
        if not isinstance(item, dict):
            return False

        # Adjust instance type
        for k, v in item.items():
            print(k, v, type(k), type(v))
            if isinstance(v, str):
                item[k] = {"S": str(v)}
            elif isinstance(v, int) or isinstance(v, float) or isinstance(v, np.integer) or isinstance(v, np.float):
                item[k] = {"N": str(v)}
            elif isinstance(v, list):
                item[k] = {"SS": v}
            elif isinstance(v, dict):
                item[k] = {"M": v}
            elif isinstance(v, bytes):
                item[k] = {"B": v}
            elif isinstance(v, bool):
                item[k] = {"BOOL": True if v is True else False}
            elif v is None:
                item[k] = {"NULL": None}
            else:
                item[k] = {"S": str(v)}

        response = self.client.put_item(
            TableName=table_name,
            Item=item,
        )
        return response
