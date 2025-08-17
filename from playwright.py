from playwright.sync_api import sync_playwright

def google_search(query):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Debugging mode
        page = browser.new_page()

        # Go to Google
        page.goto("https://www.google.com")

        # Wait for the search input field using the new ID 'APjFqb'
        try:
            page.wait_for_selector('textarea[id="APjFqb"]', timeout=10000)
            page.fill('textarea[id="APjFqb"]', query)
            page.press('textarea[id="APjFqb"]', 'Enter')

            # Wait for search results to load
            page.wait_for_selector('h3', timeout=10000)

            # Get the search result titles
            results = page.query_selector_all('h3')
            for result in results:
                print(result.inner_text())
            
            # Wait for user input to close the browser
            input("Press Enter to close the browser...")
            
        except Exception as e:
            print(f"An error occurred: {e}")

        browser.close()

# Change the query here
google_search("Playwright Python tutorial")


