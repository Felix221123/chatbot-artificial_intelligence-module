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
from multilingual import Translator
from nltk.sem import Expression
from explain_reasoning import validate_fact_with_explanation
from image_classification import CryptoLogoClassifier, classify_with_dialog



# calling the translator class
translator = Translator()

# Creating a Kernel object.
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Loading the financial q&a csv file using the panda library
qa_df = pd.read_csv("financial-qa.csv", quotechar='"', escapechar='\\')

# Loading the knowledge based file
kb = load_knowledge_base_file("knowledge_base.csv")
print(f"Knowledge base loaded with {len(kb)} finance rules.")

# creating classifier and loading image classification model
logo_clf = CryptoLogoClassifier("crypto_logo_model_tuned.h5", "labels_logo.json")


# Extra facts to make sure
extra_facts = [
    "bond(bond)",
    "stock(stock)",
    "etf(etf)",
    "bitcoin(bitcoin)",
    "interest(interest)",
    "loan(loan)",
    "inflation(inflation)",
    "deflation(deflation)",
    "recession(recession)",
    "borrower(borrower)",
    "lender(lender)",
    "prices(prices)",
    "repayment(repayment)",
    "company(company)",
    "interest(interest)",
]


# checking through the extra facts
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
    if score > 0.4:
        return qa_df.iloc[idx]['Answer']
    else:
        return "Sorry, I don't have an answer for that yet."


# translating reasoning first order logics
def translate_reasoning_preserve_fol(text_en: str, lang: str) -> str:
    """Translate headings/labels but keep the FOL part unchanged."""
    if lang == "en":
        return text_en

    lines = text_en.splitlines()
    out = []
    for line in lines:
        if line.startswith("- "):
            body = line[2:]
            if ": " in body:
                label, rest = body.split(": ", 1)
                label_t = translator.from_english(label, lang)
                out.append(f"- {label_t}: {rest}")
            else:
                out.append(line)
        else:
            out.append(translator.from_english(line, lang))
    return "\n".join(out)


# normalising yes or no questions
def normalize_yesno_is_question(raw: str) -> str:
    """
    Turns:
      'is bond a security' -> 'bond is security'
      'is future a derivative' -> 'future is derivative'
    """
    s = raw.strip().lower().rstrip("?")
    if s.startswith("is "):
        s = s[3:].strip()
    elif s.startswith("are "):
        s = s[4:].strip()

    tokens = [t for t in s.split() if t not in ("a", "an", "the")]
    if len(tokens) >= 2:
        subject = " ".join(tokens[:-1])
        pred = tokens[-1]
        return f"{subject} is {pred}"
    return raw


last_logic_fact = None       # store last checked FOL for "why?" follow-up


# Main loop
while True:
    # get user input
    try:
        userInput = input("> ")

        # detect & translate IN to English
        user_lang, userInput_en = translator.to_english(userInput)

        # SAFETY: if translation didn't change the text, don't force a non-English output language
        if user_lang != "en" and userInput_en.strip().lower() == userInput.strip().lower():
            user_lang = "en"

        txt = userInput_en.strip()
        lower = txt.lower()

        # Handle "why" / "explain" ALONE (follow-up)
        if lower in ("why", "why?", "why is that", "why is that?", "explain", "explain?"):
            if last_logic_fact:
                out_en = validate_fact_with_explanation(last_logic_fact, kb)
                print(translate_reasoning_preserve_fol(out_en, user_lang))
            else:
                msg = "I don't have anything to explain yet. Ask me to check a fact first."
                print(translator.from_english(msg, user_lang))
            continue


        # Explain mode (must happen BEFORE detect_knowledge_action)
        explain_mode = lower.startswith("why ") or lower.startswith("explain ")
        if explain_mode:
            if lower.startswith("why "):
                userInput_en = txt[4:].strip()
            else:
                userInput_en = txt[8:].strip()

        # Detect add/check action (ON THE ENGLISH TEXT)
        action, raw_fact = detect_knowledge_action(userInput_en)

        # Optional UX: treat "is X a Y" as a check even without "check that"
        if action is None and (userInput_en.lower().startswith("is ") or userInput_en.lower().startswith("are ")):
            action = "check"
            raw_fact = userInput_en

        # ADD facts to knowledge
        if action == "add":
            logic_fact = to_first_order_logic(raw_fact.lower())
            out_en = add_fact_to_knowledge_base(logic_fact, kb)
            print(translator.from_english(out_en, user_lang))
            continue


        # CHECK (explanation if explain_mode)
        elif action == "check":
            # normalize yes/no questions like "is bond a security"
            if raw_fact.lower().startswith(("is ", "are ")):
                raw_fact = normalize_yesno_is_question(raw_fact)

            logic_fact = to_first_order_logic(raw_fact.lower())
            last_logic_fact = logic_fact  # store for follow-up "why?"

            if explain_mode:
                out_en = validate_fact_with_explanation(logic_fact, kb)
                print(translate_reasoning_preserve_fol(out_en, user_lang))
            else:
                out_en = validate_fact(logic_fact, kb)
                print(translator.from_english(out_en, user_lang))
            continue

    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break


    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'

    out_en = ""

    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput_en.upper())

    # post-process the answer for commands
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            out_en = params[1]
            print(translator.from_english(out_en, user_lang))
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=True)
                out_en = wSummary
            except:
                out_en = "Sorry, I do not know that. Be more specific!"
        elif cmd == 2:
            out_en = classify_with_dialog(logo_clf, open_preview=True)

        elif cmd == 99:
            # AIML couldn't handle the question → use cosine similarity
            out_en = find_similar_question(preprocess_input(userInput_en))
        else:
            out_en = "I did not get that, please try again."
    else:
        if not answer or not answer.strip():
            out_en = find_similar_question(userInput_en)
        else:
            out_en = answer

    # translate OUT to user language
    print(translator.from_english(out_en, user_lang))
