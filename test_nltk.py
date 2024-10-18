# test_nltk.py
import nltk
nltk.download('punkt')

text = "Hello world. This is a test."
sentences = nltk.sent_tokenize(text)
print(sentences)

