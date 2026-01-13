# PromptCart Fast API Architecture 
This project folder uses the python aiml library and wikipedia to create a chatbot using information from the xml sheet

## How to start project
- `pip(windows)/pip3(mac) install -r requirements.txt`
- `python3 mybot-basic.py`







## Documentation







#### Extra functionalities
   (Extra Functionalities for Part A)Added multilingual interaction using NLP language detection and translation. Non-English user input is translated to English before AIML/TF-IDF processing, and responses are translated back to the user’s language, extending Task-A without replacing the required components.

   (Extra Functionalities for Part B)The FOL reasoning component was extended with explanation-based inference: when a query is proven true, the system returns the supporting fact(s) and the rule chain used to derive the conclusion