import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def get_all_assessment_links():
    links = set()

    start = 0

    while True:
        url = f"{CATALOG_URL}?start={start}"
        print(f"Scraping catalog page starting at {start}...")

        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        cards = soup.select("a[href*='/products/product-catalog/view/']")

        if not cards:
            break

        for card in cards:
            href = card.get("href")

            # filter only individual test solutions
            if "/view/" in href.lower():

                unwanted = [
                    "job-profiling",
                    "framework",
                    "guide",
                    "interview",
                    "pre-packaged"
                ]

                if not any(word in href.lower() for word in unwanted):
                    full_url = BASE_URL + href
                    links.add(full_url)

        start += 12
        time.sleep(1)

    return list(links)


def scrape_assessment(url):
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        # name
        title_tag = soup.find("h1")
        name = title_tag.text.strip() if title_tag else ""

        # description
        meta = soup.find("meta", {"name": "description"})
        description = meta["content"].strip() if meta else ""

        return {
            "name": name,
            "url": url,
            "description": description
        }

    except Exception as e:
        print(f"Error scraping {url}")
        return None


def main():
    print("Getting all assessment links...")
    links = get_all_assessment_links()

    print(f"Total links found: {len(links)}")

    data = []

    for i, link in enumerate(links):
        print(f"Scraping {i+1}/{len(links)}")

        result = scrape_assessment(link)

        if result and result["name"]:
            data.append(result)

        time.sleep(0.5)

    df = pd.DataFrame(data)

    df.drop_duplicates(inplace=True)

    df.to_csv("data/assessments.csv", index=False)

    print("\nDONE")
    print(f"Total assessments saved: {len(df)}")
    print("Saved to data/assessments.csv")


if __name__ == "__main__":
    main()