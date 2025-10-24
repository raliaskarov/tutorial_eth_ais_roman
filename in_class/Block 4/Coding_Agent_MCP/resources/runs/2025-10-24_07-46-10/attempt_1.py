import requests

url = "https://upload.wikimedia.org/wikipedia/en/7/73/Trollface.png"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("plot.png", "wb") as file:
        file.write(response.content)