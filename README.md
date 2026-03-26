# AI CHATBOT (ARTIFICIAL INTELLIGENCE MODULE)
This project folder uses the python aiml library and wikipedia to create a chatbot using information from the xml sheet

## How to start project
- `pip(windows)/pip3(mac) install -r requirements.txt`
- `python3 mybot-basic.py`








#### Extra functionalities
   (Extra Functionalities for Part A)Added multilingual interaction using NLP language detection and translation. Non-English user input is translated to English before AIML/TF-IDF processing, and responses are translated back to the user’s language, extending Task-A without replacing the required components.

   (Extra Functionalities for Part B)The First Order Logic(FOL) reasoning component was extended with explanation-based inference: when a query is proven true, the system returns the supporting fact(s) and the rule chain used to derive the conclusion

   (Extra Functionalities for Part C)Hyper-parameter optimisation (tuning) for Convolutional Neural Networks(CNN) (MobileNetV2 transfer learning)
   Showing the baseline accuracy vs tuned accuracy, plus the best parameters found.
   You started with a MobileNetV2 transfer learning classifier (baseline).
   For the extra task, you applied hyperparameter optimisation using Random Search (KerasTuner).
   You tuned dropout + learning rate, then fine-tuned the last 30 layers.
   You compared baseline test accuracy vs tuned test accuracy.
   Explain why accuracy changed:
   dropout reduces overfitting on small datasets
   learning rate affects convergence stability
   fine-tuning improves feature adaptation to crypto logos
