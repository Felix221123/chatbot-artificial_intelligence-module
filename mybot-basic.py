#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from format import preprocess_input, detect_knowledge_action, to_first_order_logic
from rule_base import load_knowledge_base_file, add_fact_to_knowledge_base,validate_fact


# Create a Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Loading the financial q&a csv file using the panda library
qa_df = pd.read_csv("financial-qa.csv", quotechar='"', escapechar='\\')

# Loading the knowledge based file 
kb = load_knowledge_base_file("knowledge_base.csv")
print(f"Knowledge base loaded with {len(kb)} finance rules.")

# Extra facts to make sure 
extra_facts = [
    "bond(bond)",
    "stock(stock)",
    "etf(etf)",
    "bitcoin(bitcoin)",
    "cryptocurrency(bitcoin)"
]
from nltk.sem import Expression
read_expression = Expression.fromstring
for f in extra_facts:
    kb.append(read_expression(f))



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


# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
        action, raw_fact = detect_knowledge_action(userInput)

        # checking if user attempt wants to add knowledge 
        if action == "add":
            logic_fact = to_first_order_logic(raw_fact.lower())
            print(add_fact_to_knowledge_base(logic_fact, kb))
            continue

        elif action == "check":
            logic_fact = to_first_order_logic(raw_fact.lower())
            print(validate_fact(logic_fact, kb))
            continue

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