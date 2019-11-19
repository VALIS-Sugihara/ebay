import boto3
import numpy as np


class Dynamodb():
    region_name = "us-east-1"

    def __init__(self, table_name=None):
        self.resource = boto3.resource("dynamodb", region_name=self.region_name)
        if table_name is not None:
            self.set_table(table_name)
        self.client = boto3.client('dynamodb', region_name=self.region_name)

    def set_table(self, table_name):
        self.table = self.resource.Table(table_name)

    def get_item(self, key={}):
        # Check Key Schema
        primary_keys = [schema["AttributeName"] for schema in self.table.key_schema]
        for pk in primary_keys:
            if pk not in key.keys():
                return False

        response = self.table.get_item(Key=key)
        return response

    def query(self, **kwargs):
        response = self.table.query(**kwargs)
        return response

    # def put_item(self, items):
    #     if not isinstance(items, dict):
    #         return False
    #
    #     response = self.table.put_item(Item=items)
    #     return response

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
