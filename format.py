# this file contains functions to format certain instructions, rules and logics in the chatbot




# packages needed
import re


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




# function detects whether a user input is an attempt to check a fact or add knowledge
def detect_knowledge_action(user_input: str):
    """
    Detects if user input is an attempt to add or check knowledge.
    Returns a tuple (action_type, cleaned_text)
    where action_type is 'add', 'check', or None
    """

    # Normalize input
    text = user_input.strip().lower()

    # Variations for adding new knowledge
    know_triggers = [
        "i know that",
        "he knows that",
        "she knows that",
        "they know that",
        "we know that",
        "remember that"
    ]

    # Variations for checking knowledge
    check_triggers = [
        "check that",
        "verify that",
        "confirm that",
        "prove that",
        "is it true that",
        "check whether"
    ]

    # Check for knowledge-add triggers
    for trigger in know_triggers:
        if text.startswith(trigger):
            cleaned = text[len(trigger):].strip()
            return ("add", cleaned)

    # Check for knowledge-check triggers
    for trigger in check_triggers:
        if text.startswith(trigger):
            cleaned = text[len(trigger):].strip()
            return ("check", cleaned)

    # Not a knowledge command
    return (None, None)


# function converts sentences to first order logic for fact checking
def to_first_order_logic(statement: str):
    """
    Convert a simple English-like statement into First Order Logic.
    Handles pattern: 'X is Y' (optionally with 'a/an/the'),
    mapping to: predicate(subject), all lowercase.

    Examples:
        'bitcoin is security'       -> 'security(bitcoin)'
        'bond is a security'        -> 'security(bond)'
        'apple is a company'        -> 'company(apple)'

    If no ' is ' is found, assumes the user already wrote FOL
    and just returns the stripped, lowercased string.
    """
    text = statement.strip().lower().rstrip(".")

    if " is " not in text:
        # Assume it's already FOL or something more complex
        return text

    left, right = text.split(" is ", 1)

    subject = left.strip().replace(" ", "_")

    # Strip articles from the predicate part
    right_tokens = [t for t in right.split() if t not in ("a", "an", "the")]
    if not right_tokens:
        return text 

    predicate = right_tokens[0]  

    return f"{predicate.lower()}({subject.lower()})"
