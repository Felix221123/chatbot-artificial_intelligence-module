#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Create a Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Loading the csv file using the panda library
qa_df = pd.read_csv("financial-qa.csv", quotechar='"', escapechar='\\')


# Welcome user
print('')
print('\n***********  Welcome to this chat bot. Please feel free to ask questions from me!  ***********\n')

# Ask user their name
user_name = input("Would you be my friend? Please tell me what your name is: ")
print(f'Hello {user_name}')

print('\n***********  Welcome to your Financial Chatbot, ' + user_name + '!,what do you want to know?  ***********\n')


# Prepare TF-IDF model (lowercase everything)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(qa_df['Question'].str.lower())  

# using the similarity cosine rule for text matching
def find_similar_question(user_input):
    user_vec = vectorizer.transform([user_input.lower()])
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    score = similarities[0, idx]
    if score > 0.4:  # Adjust threshold as needed
        return qa_df.iloc[idx]['Answer']
    else:
        return "Sorry, I don't have an answer for that yet."

# function to preprocess text before vectorizing to find a match
def preprocess_input(text):
    text = text.lower()

    # Common filler words and phrases that add no real meaning
    stop_phrases = [
        "do you know", "can you tell me", "please", "could you", "would you", "i want to know",
        "can you explain", "tell me about", "give me info about", "give me information about",
        "explain", "let me know", "kindly", "do you think", "what can you say about",
        "i would like to know", "i'm curious about", "can you describe"
    ]

    stop_words = [
        "a", "an", "the", "then", "of", "is", "are", "was", "were",
        "and", "or", "in", "on", "at", "for", "to", "from", "by", "about",
        "that", "this", "those", "these", "it", "its", "me", "my", "you",
        "your", "yours", "our", "us", "we", "they", "them", "their", "there",
        "what", "who", "when", "where", "how", "why", "does", "do", "did",
        "will", "would", "should", "could", "can", "may"
    ]

    # Remove whole filler phrases first (important to do before single words)
    for phrase in stop_phrases:
        text = text.replace(phrase, " ")

    # Then remove individual common stop words
    pattern = r'\b(' + '|'.join(stop_words) + r')\b'
    text = re.sub(pattern, '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput.upper())
    #post-process the answer for commands
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=True)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 99:
            # AIML couldn't handle the question → use cosine similarity
            fallback = find_similar_question(preprocess_input(userInput))
            print(fallback)
        else:
            print("I did not get that, please try again.")
    else:
            # If AIML gives no result or empty response → fallback to cosine similarity
            if not answer.strip():
                fallback = find_similar_question(userInput)
                print(fallback)
            else:
                print(answer)