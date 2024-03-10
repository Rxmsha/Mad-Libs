import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.lm import MLE
from nltk.util import everygrams


#preprocessing data 
def preprocess_data(text):
  #remove html tags
  text = re.sub('<.*?>', '', text)

  #convert text to lowercase
  text = text.lower()

  #remove punctuation
  text = re.sub(f'[^\w\s]', '', text)

  #remove stop words
  stop_words = set(stopwords.words())

  #tokenizing text - breaking sentence into words for faster labelling to parts of speech
  tokens = word_tokenize(text)

  #array of all of our words in the text
  text = [word for word in tokens if word not in stop_words] 


#nltk installs
nltk.download('punkt')


#language model function
def language_model(text_data):

    # tokens is a list of lists, where each list is a sentence in the text data - each item in the sub-list is a word in the sentence
    
    #[[sentence1], [sentence2], [sentence3], [sentence4], [sentence5], [sentence6], [sentence7]] = text_data

    tokens = [word_tokenize(sentence) for sentence in text_data]
    tokens = [word for sublist in tokens for word in sublist]
    bigrams = list(everygrams(tokens, max_len=2))

    if bigrams: 
        language_model = MLE(2)

        vocabulary = nltk.lm.Vocabulary(tokens) 
        language_model.fit([bigrams], vocabulary)
        return language_model

    else: 
        print("No bigrams found. Please input valid text data.")
        return None

text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "The cat in the hat plays with the dog in the fog.",
    "To be or not to be, that is the question.",
    "All's well that ends well.",
    "A stitch in time saves nine.",
    "The early bird catches the worm.",
    "Two households, both alike in dignity\nIn fair Verona, where we lay our scene\nFrom ancient grudge break to new mutiny\nWhere civil blood makes civil hands unclean.",
    "From forth the fatal loins of these two foes\nA pair of star-crossed lovers take their life;",
    "Whose misadventured piteous overthrow \n Doth with their death bury their parents’ strife.",
    "The fearful passage of their death-marked love\nAnd the continuance of their parents’ rage,\nWhich, but their children’s end, naught could remove,\nIs now the two hours’ traffic of our stage;"
]

language_model = language_model(text_data)


#generate mad libs
def generate_mad_libs(template, language_model):
  placeholders = re.findall(r'\{(.*?)\}', template)
  mad_libs = template

  for i in placeholders:
    if i == "noun":
      replacement = input("Enter a noun: ")
    elif i == "verb":
      replacement = input("Enter a verb: ")
    elif i == "adjective":
      replacement = input("Enter a adjective: ")
    elif i == "adverb":
      replacement = input("Enter a adverb: ")
    else:
      replacement = input("Enter a randoom word: ")
    
    mad_libs = mad_libs.replace(f"{{{i}}}", replacement, 1)
  
  return mad_libs

template = "The {adjective} {noun} {verb} {adverb} through the {adjective} forest. Then, all of a sudden, the {adjective} {noun} {adverb} {verb}! It was so {adjective} that {noun} {adverb} {verb} away. Written by {noun} and {noun}."

shakespeare_template = "Two {noun}, both alike in {adjective}\nIn {adjective} Verona, where we {verb} our {noun}\nFrom {adjective} grudge {verb} to {adjective} mutiny\nWhere {adjective} blood {verb} civil {noun} {adjective}."

#finished_mad_libs = generate_mad_libs(template, language_model)
finished_mad_libs = generate_mad_libs(shakespeare_template, language_model)

print(finished_mad_libs)