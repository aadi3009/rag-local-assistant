import requests

# SearxNG instance URL
url = "http://localhost:8080/search"

# Query parameters
params = {
    'q': 'what is vyvance?',
    'format': 'json'
}

# Set headers to mimic a browser request
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
#     'Accept': 'application/json'
# }
# headers = {
#     'User-Agent': 'curl/7.64.1',
# } 

# response = requests.get(url, params=params, headers=headers)

# Try without custom headers first
response = requests.get(url, params=params)

if response.status_code == 403:
    print("403 Forbidden: The request was blocked.")
else:
    print(response.json())
