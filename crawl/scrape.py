import argparse
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_rows_from_url(url: str):
    # Set up headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send a GET request
    response = requests.get(url, headers=headers)

    # Check for successful response
    if response.status_code != 200:
        raise ConnectionError(f"Network error: {response.status_code}")

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table or list that contains the deck data
    # (This may need adjustment based on the actual structure of the page)
    deck_rows = soup.find_all('tr')  # Example: rows in a table

    return deck_rows


def save_image_from_url(url: str, save_dir: str, filename: str = None, prefix: str = ""):
    """
    Save an image from a URL to a local folder.

    Args:
        url (str): The URL of the image.
        save_dir (str): The directory to save the image.
        filename (str): Optional. The name of the file to save. If not provided, use the name from the URL.
        prefix (int): Optional. The prefix of the image. If not provided, "".
    """
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract filename from URL if not provided
    if filename is None:
        filename = url.split("/")[-1]

    # Full path to save the file
    save_path = os.path.join(save_dir, f"{prefix}{filename}")

    try:
        # Get the image content
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for failed requests

        # Write the content to a local file
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        # print(f"Image saved successfully: {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")


def get_rank_data(rank_url: str,
                  output_dir: str,
                  use_match_url: bool = True,
                  ):
    # Extract relevant data (assuming decks are listed in table rows)
    decks = []
    deck_name_to_limit_id = {}
    deck_name_to_limit_url = {}

    deck_rows = get_rows_from_url(rank_url)

    for row in deck_rows:
        cells = row.find_all('td')
        if cells:
            root_url = "https://play.limitlesstcg.com"
            match_url = cells[2].find('a')['href'] if cells[2].find('a') else ""
            deck_url_name = match_url.split("/")[-1].split("?")[0]
            match_url = match_url.replace("?format", "/matchups/?format")
            match_url = root_url + match_url

            if deck_url_name != "" and use_match_url:  # goto the match-up page for this deck
                get_match_data(match_url, output_dir, save_fn=f"{deck_url_name}.csv")

            # Extract information from each cell
            deck = {
                "Rank": cells[0].get_text(strip=True) if len(cells) > 0 else "",
                "Deck": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                "Count": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                "Score": cells[5].get_text(strip=True) if len(cells) > 5 else "",
                "Win%": cells[6].get_text(strip=True) if len(cells) > 6 else "",
            }
            deck_name = deck["Deck"]
            deck_name_to_limit_id[deck_name] = deck_url_name
            deck_name_to_limit_url[deck_name] = match_url
            decks.append(deck)

            # Save figures of decks
            fig_urls = [c['src'] for c in cells[1].find_all('img')]
            for fig_idx, fig_url in enumerate(fig_urls):
                save_image_from_url(
                    fig_url,
                    os.path.join(output_dir, "fig", deck_url_name),
                    prefix=f"{fig_idx}_"
                )

    # Append deck_url into decks
    for deck in decks:
        deck_name = deck["Deck"]
        deck["Limit_ID"] = deck_name_to_limit_id[deck_name]
        deck["Limit_URL"] = deck_name_to_limit_url[deck_name]

    # Save the data to a DataFrame
    df = pd.DataFrame(decks)

    # Export to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    save_fn = os.path.join(output_dir, "00_rank_data.csv")
    df.to_csv(save_fn, index=False)
    print(f"Rank data saved to {save_fn}")

    return df


def get_match_data(match_url: str,
                   output_dir: str,
                   save_fn: str,
                   ) -> pd.DataFrame:
    # Extract relevant data (assuming decks are listed in table rows)
    decks = []
    deck_rows = get_rows_from_url(match_url)

    for row in deck_rows:
        cells = row.find_all('td')
        if cells:
            # Extract information from each cell
            deck = {
                "Opponent": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                "Matches": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                "Score": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                "Win%": cells[4].get_text(strip=True) if len(cells) > 4 else "",
            }
            decks.append(deck)

    # Save the data to a DataFrame
    df = pd.DataFrame(decks)

    # Export to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_fn)
    df.to_csv(save_path, index=False)
    print(f"Match-Up data saved to {save_path}")

    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser("Crawling decks data from limitless webpage.")
    args.add_argument("-u", "--url", type=str,
                      default="https://play.limitlesstcg.com/decks?format=standard&rotation=2022&set=SIT",
                      help="URL of Limitless.")
    args.add_argument("--output_dir", type=str, default="crawl/output", help="Output directory.")
    args.add_argument("--no_match_url", action="store_true", default=False,
                      help="If True: not iterate sub match-up urls.")
    args.add_argument("--child_url", action="store_true",
                      help="If True: the provided url is a match-up url")
    args = args.parse_args()

    if not args.child_url:  # default
        get_rank_data(args.url, args.output_dir,
                      use_match_url=not args.no_match_url,
                      )
    else:
        get_match_data(args.url, args.output_dir, save_fn="tmp.csv")
