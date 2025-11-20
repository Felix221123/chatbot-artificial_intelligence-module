#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise Natural language toolkit package

from nltk.sem import Expression
from nltk.inference import ResolutionProver
import csv

# creating a parser for first order logic
read_expression = Expression.fromstring


# Loading the knowledge based csv file
def load_knowledge_base_file(file_path="knowledge_base.csv"):
    knowledge_base = []
    read_expression = Expression.fromstring

    try:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                expr_str = row.get("Expression", "").strip().strip('"')
                if not expr_str:
                    continue
                expr_str = expr_str.lower()
                try:
                    expr = read_expression(expr_str)
                    knowledge_base.append(expr)

                    # Auto-add self facts for unary predicates in rules
                    if "->" in expr_str and "(x)" in expr_str:
                        left_side = expr_str.split("->")[0].strip()
                        predicate = left_side.split("(")[0]
                        try:
                            knowledge_base.append(read_expression(f"{predicate}({predicate})"))
                        except:
                            pass
                except Exception as e:
                    print(f"Invalid expression in knowledge_base: {expr_str} -> {e}")

    except FileNotFoundError:
        print(f"⚠️ Knowledge base file '{file_path}' not found.")

    return knowledge_base




# Adding a new fact 
def add_fact_to_knowledge_base(statement, knowledge_base):
    """
    Adds a new fact to the Knowledge Base if it doesn't contradict existing facts.
    Example: "Company(Google)" or "Bond(x) -> Security(x)"
    """
    try:
        expression = read_expression(statement)
        contradiction = ResolutionProver().prove(None, assumptions=knowledge_base + [expression])
        if contradiction:
            return "That contradicts what I already know!"
        else:
            knowledge_base.append(expression)
            return "OK, I will remember that."
    except Exception as e:
        return f"Sorry, I couldn’t understand that statement ({e})."


# checking fact validity
def validate_fact(statement, knowledge_base):
    """
    Checks if a statement logically follows from the knowledge Base.
    Returns: 'Correct', 'Incorrect', or 'I don’t know'
    """
    try:
        expression = read_expression(statement)
        result = ResolutionProver().prove(expression, assumptions=knowledge_base)
        if result:
            return "Correct."
        else:
            negation = read_expression(f"-({statement})")
            if ResolutionProver().prove(negation, assumptions=knowledge_base):
                return "Incorrect."
            else:
                return "I don’t know."
    except Exception as e:
        return f"Sorry, I couldn’t process that check ({e})."