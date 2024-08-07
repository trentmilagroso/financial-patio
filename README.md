Still trying to figure out how to make this more accurate, as right now, the cross-validation score is at a 52%.

Uses random forest classifier instead of linear regression as stock is volatile. Instead of linear relationships, it is easier for stocks to use random forest to create however many decision trees and takes the average of X to output Y.

This is a project that I used to teach me fundamentals on machine learning, specifically about random forests and decision trees.

August 07, 2024
Added a webscraper that goes through Yfinance in search of whatever stock you input. Returns a sentiment value for the news articles and indicates whether or not you should buy, hold, or sell the stock.
- Also trained and fine-tuned my own BERT model that uses publicly available data (FinancialPhraseBank) for better results. 
