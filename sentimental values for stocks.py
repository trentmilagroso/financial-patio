from playwright.sync_api import sync_playwright
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

def get_articles_playwright(ticker):
    with sync_playwright() as p:
        # Launch a headless browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Yahoo Finance page for the specified ticker
        search_url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        page.goto(search_url)

        # Wait for the articles to load
        page.wait_for_selector('a.subtle-link.fin-size-small.titles.noUnderline.yf-13p9sh2')

        # Extract articles
        articles = page.query_selector_all('a.subtle-link.fin-size-small.titles.noUnderline.yf-13p9sh2')

        if not articles:
            print("No articles found. Please check the ticker and try again.")
            browser.close()
            return []

        # Extract titles and links
        article_data = []
        for article in articles:
            title = article.get_attribute('title')
            link = article.get_attribute('href')
            article_data.append((title, link))

        # Close the browser
        browser.close()
        return article_data  # Return list of article titles and links

def fetch_article_content(link):
    # Fetch the article content
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract paragraphs
    paragraphs = soup.find_all('p')
    content = ' '.join([para.text for para in paragraphs])
    
    # Summarize the content (first 500 characters for simplicity)
    summary = content[:500]
    
    return summary, content  # Return summary and full content

def analyze_sentiment(text):
    # Analyze sentiment with TextBlob
    text_blob_analysis = TextBlob(text)
    text_blob_sentiment = text_blob_analysis.sentiment.polarity
    
    # Analyze sentiment with VADER
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = vader_analyzer.polarity_scores(text)
    vader_sentiment_score = vader_sentiment['compound']  # Use the compound score for VADER

    # Combine scores (average for simplicity)
    average_sentiment = (text_blob_sentiment + vader_sentiment_score) / 2
    
    return average_sentiment

def main():
    while True:
        ticker = input("Enter a stock ticker: ").strip().upper()
        articles = get_articles_playwright(ticker)

        if articles:
            for title, link in articles:
                summary, content = fetch_article_content(link)
                sentiment = analyze_sentiment(content)
                print(f"Title: {title}\nLink: {link}\nSummary: {summary}\nSentiment Score: {sentiment:.2f}\n")
            break  # Exit the loop if articles are found and processed
        else:
            print("Please try again with a valid stock ticker.")

if __name__ == "__main__":
    main()
    