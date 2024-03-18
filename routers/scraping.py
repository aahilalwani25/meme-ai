# In views.py of your web_scraper app
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.request import Request
from bs4 import BeautifulSoup

@csrf_exempt
def get_funny_content(request: Request, query):
    # print(query)
    # Quora
    quora_url = f"https://www.quora.com/search?q={query}"
    quora_response = requests.get(quora_url)
    quora_soup = BeautifulSoup(quora_response.text, 'html.parser')
    quora_results = quora_soup.find_all('div', class_='pagedlist_item')

    # Reddit
    reddit_url = f"https://www.reddit.com/search/?q={query}"
    reddit_response = requests.get(reddit_url, headers={'User-Agent': 'Mozilla/5.0'})
    reddit_soup = BeautifulSoup(reddit_response.text, 'html.parser')
    reddit_results = reddit_soup.find_all('div', class_='scrollerItem')

    # Another site for memes
    # You need to replace 'another_site_url' with the actual URL of the site you want to scrape for memes
    another_site_url = f"https://example.com/search?q={query}"
    another_site_response = requests.get(another_site_url)
    another_site_soup = BeautifulSoup(another_site_response.text, 'html.parser')
    another_site_results = another_site_soup.find_all('div', class_='meme')

    # Extracting data from results
    quora_content = [result.text.strip() for result in quora_results]
    reddit_content = [result.text.strip() for result in reddit_results]
    another_site_content = [result.text.strip() for result in another_site_results]

    # Combining results
    all_content = {
        "Quora": quora_content,
        "Reddit": reddit_content,
        "Another Site": another_site_content
    }

    return JsonResponse(all_content)
