# this file contains functions to format certain instructions, rules and logics in the chatbot




# packages needed
import re


_TYPO_FIX = {
    "infaltion": "inflation",
    "crytocurrency": "cryptocurrency",
}

_PLURAL_MAP = {
    "bonds": "bond",
    "stocks": "stock",
    "securities": "security",
    "prices": "prices",
}



def _fix_typos(text: str) -> str:
    for bad, good in _TYPO_FIX.items():
        text = re.sub(rf"\b{re.escape(bad)}\b", good, text)
    return text

def _clean_entity(phrase: str) -> str:
    # remove articles + join underscores
    tokens = [t for t in phrase.strip().lower().split() if t not in ("a", "an", "the")]
    if not tokens:
        return phrase.strip().lower().replace(" ", "_")

    head = tokens[0]
    head = _PLURAL_MAP.get(head, head)
    return head.replace(" ", "_")




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



def _singularize(word: str) -> str:
    w = word.strip().lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w




# function converts sentences to first order logic for fact checking
def to_first_order_logic(statement: str) -> str:
    """
    Supports:
      - X is Y -> y(x)
      - X verb Y -> verb(x,y)
    """
    text = statement.strip().lower().rstrip(".?!")
    text = _fix_typos(text)

    # Passive patterns with spaces: "interest is paid by borrower"
    m = re.match(r"^(.+?)\s+is\s+paid\s+by\s+(.+)$", text)
    if m:
        left, right = m.groups()
        return f"paidby({_clean_entity(left)},{_clean_entity(right)})"

    m = re.match(r"^(.+?)\s+is\s+paid\s+to\s+(.+)$", text)
    if m:
        left, right = m.groups()
        return f"paidto({_clean_entity(left)},{_clean_entity(right)})"

    # 1) "X is Y" OR "X is <binaryverb> Y"
    if " is " in text:
        left, right = text.split(" is ", 1)
        subj = _clean_entity(left)

        right_tokens = [t for t in right.split() if t not in ("a", "an", "the")]
        if not right_tokens:
            return text

        pred = right_tokens[0]
        pred = _PLURAL_MAP.get(pred, pred)

        # special case: "interest is paidby a borrower" / "interest is paidto a lender"
        if pred in ("paidby", "paidto") and len(right_tokens) >= 2:
            obj = _PLURAL_MAP.get(right_tokens[1], right_tokens[1])
            return f"{pred}({subj},{_clean_entity(obj)})"

        # normal unary: "bond is security"
        return f"{pred}({subj})"

    # 2) Binary: "X verb Y"
    m = re.match(r"^(.+?)\s+([a-z_]+)\s+(.+)$", text)
    if m:
        subj_raw, verb, obj_raw = m.groups()
        subj = _clean_entity(subj_raw)
        obj = _clean_entity(obj_raw)

        # map "borrower pays interest" -> paidby(interest, borrower)
        if verb in ("pay", "pays", "paid"):
            return f"paidby({obj},{subj})"

        # map "lender receives interest" -> paidto(interest, lender)
        if verb in ("receive", "receives", "get", "gets"):
            return f"paidto({obj},{subj})"

        return f"{verb}({subj},{obj})"

    return text
