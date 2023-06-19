
import requests

year = 2022
month = 3

message = {
    'year':2022,
    'month':4
    }

url = "http://localhost:9696/app"
response = requests.post(url, json=message)
print(response.json())


# print('starting')
# print(predict(year, month))