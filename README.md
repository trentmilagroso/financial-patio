Still trying to figure out how to make this more accurate, as right now, the cross-validation score is at a 52%.

Uses random forest classifier instead of linear regression as stock is volatile. Instead of linear relationships, it is easier for stocks to use random forest to create however many decision trees and takes the average of X to output Y.

This is a project that I used to teach me fundamentals on machine learning, specifically about random forests and decision trees.

Currently working on adding a web scraper that takes news and assigns the title and article with a sentiment value for stock. If the predictor says to invest, yet the sentiment value shows negative for the stock, it would probably be best not to invest.
