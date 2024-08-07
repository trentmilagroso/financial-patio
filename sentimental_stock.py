from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# Load the fine-tuned model and tokenizer
finetuned_model_path = './finetuned_finbert'  # Update the path if needed
finetuned_tokenizer = BertTokenizer.from_pretrained(finetuned_model_path)
finetuned_model = BertForSequenceClassification.from_pretrained(finetuned_model_path)
finetuned_pipeline = pipeline('sentiment-analysis', model=finetuned_model, tokenizer=finetuned_tokenizer)

# List of phrases to ignore
irrelevant_phrases = [
    "Thank you for your patience.",
    "Our engineers are working quickly to resolve the issue.",
    "Read the full article on",
    "Click here to read more",
    "For more information",
    "This article was originally published on",
    "Sign up for our newsletter",
    "Click here to view the full article"
]
# Uses Playwright, a headless browser, to fetch articles from YFINANCE
def get_articles_playwright(ticker):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        search_url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        page.goto(search_url)
        page.wait_for_selector('a.subtle-link.fin-size-small.titles.noUnderline.yf-13p9sh2')
        articles = page.query_selector_all('a.subtle-link.fin-size-small.titles.noUnderline.yf-13p9sh2')
        
        #Are there articles
        if not articles:
            print("No articles found. Please check the ticker and try again.")
            browser.close()
            return []
        
        # Create a list to hold articles
        article_data = []
        for article in articles:
            title = article.get_attribute('title')
            link = article.get_attribute('href')
            if link.startswith('/'):
                link = f'https://finance.yahoo.com{link}'
            article_data.append((title, link))

        browser.close()
        return article_data # return list

# Fetch the contents of the article
def fetch_article_content(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([para.text for para in paragraphs])
    summary = content[:500]
    return summary, content

# Filter out irrelevant phrases
def filter_irrelevant_content(text):
    for phrase in irrelevant_phrases:
        if phrase in text:
            return False
    return True

# Using the fine tuned model, this will analyze the sentiment
def analyze_sentiment_finetuned(text):
    sentences = text.split('.')
    sentiment_scores = []

# Analyze through each sentence
    for sentence in sentences:
        if sentence.strip() and filter_irrelevant_content(sentence):
            inputs = finetuned_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512) # Essential as BERT can only hold up to 512
            sentiments = finetuned_model(**inputs)
            probs = sentiments.logits.softmax(dim=-1)

            sentiment_values = {'negative': -1, 'neutral': 0, 'positive': 1}
            avg_score = sum((sentiment_values[label] * probs[0][i].item()) for i, label in enumerate(sentiment_values.keys())) / len(sentiment_values)
            sentiment_scores.append(avg_score)
    # Return average score if computed
    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores)
    else:
        return 0  # Default neutral if no sentences were analyzed

# Should you trade
def suggest_action(sentiment_score):
    if sentiment_score > 0.3:
        return "Buy"
    elif sentiment_score < -0.3:
        return "Sell"
    else:
        return "Hold"

# Main function that analyzes the articles
def main():
    while True:
        ticker = input("Enter a stock ticker: ").strip().upper() # Input stock
        articles = get_articles_playwright(ticker)

        if articles:
            for title, link in articles:
                summary, content = fetch_article_content(link)
                
                if not filter_irrelevant_content(summary):
                    print(f"Title: {title}\nLink: {link}\nIrrelevant content detected. Skipping analysis.\n")
                    continue

                sentiment_score = analyze_sentiment_finetuned(content)
                action = suggest_action(sentiment_score)
                
                print(f"Title: {title}\nLink: {link}\nSummary: {summary}\nSentiment Score: {sentiment_score:.2f}\nSuggested Action: {action}\n")
            break
        else:
            print("Please try again with a valid stock ticker.")

if __name__ == "__main__":
    main()