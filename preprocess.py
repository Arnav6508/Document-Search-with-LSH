import nltk
import re
import string

nltk.download('stopwords')

from nltk.corpus import stopwords         
from nltk.stem import PorterStemmer        
from nltk.tokenize import TweetTokenizer 

def regular_expression(tweet):
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    return tweet

def tokenize(tweet):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    
    return tokenizer.tokenize(tweet)

def remove_stopwords(tokenized_tweet):
    stopwords_eng = stopwords.words('english') 
    punctuations = string.punctuation

    words = []
    for word in tokenized_tweet:
        if word not in stopwords_eng and word not in punctuations: words.append(word)
    return words

def stemming(tokenized_tweet):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokenized_tweet]

def preprocess(tweet):
    tweet = regular_expression(tweet)
    tweet = tokenize(tweet)
    tweet = remove_stopwords(tweet)
    tweet = stemming(tweet)
    return tweet