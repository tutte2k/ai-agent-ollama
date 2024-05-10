"""
read the contents of test.py and write a python script that calls the post endpoint to make a new item
"""
import requests


def create_item(api_url, item_data):
    response = requests.post(api_url, json=item_data)
    return response.json()
