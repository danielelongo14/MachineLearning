
#Imports

from textblob import TextBlob
from newspaper import Article
import nltk
nltk.download("punkt")

url = "https://en.wikipedia.org/wiki/Classical_architecture"
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary
print(text)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity #from -1 to 1
print(f"The sentiment of this text is {sentiment}")


with open("C:/Users/danie/PycharmProjects/TextAnalysis/venv/mytext.txt","r") as f:
    text = f.read()

print(text)
blob = TextBlob(text)
sentiment = blob.sentiment.polarity  # from -1 to 1
print(f"The sentiment of this text is {sentiment}")