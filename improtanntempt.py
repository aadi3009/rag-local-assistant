import requests

# Function to send a search query to SearXNG API and return results
def search_searxng(query):
    url = "http://localhost:8080/search"
    params = {
        'q': query,      # The search query
        'format': 'json'  # Request the results in JSON format
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return the parsed JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error querying SearXNG: {e}")
        return None

# Example usage
if __name__ == "__main__":
    query = "example search query"  # Replace with your search term
    results = search_searxng(query)

    if results:
        print("Search Results:")
        for result in results.get('results', []):
            print(f"- {result.get('title')}: {result.get('url')}")
    else:
        print("No results found or there was an error.")
