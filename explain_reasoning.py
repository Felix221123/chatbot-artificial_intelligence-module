# explain_reasoning.py
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.sem.logic import (
    ImpExpression,
    ApplicationExpression,
    IndividualVariableExpression,
    Variable,
)


read_expression = Expression.fromstring



def _a_or_an(word: str) -> str:
    return "an" if word[:1].lower() in ("a", "e", "i", "o", "u") else "a"


def _prettify_token(tok: str) -> str:
    return tok.replace("_", " ").strip()


def _is_proper_noun(tok: str) -> bool:
    # add more if you want
    return tok in {"bitcoin", "ethereum", "uk", "usa", "sec"}


def _expr_to_english_sentence(expr) -> str:
    """
    security(bond) -> "A bond is a security."
    cryptocurrency(bitcoin) -> "Bitcoin is a cryptocurrency."
    issues(uk,bond) -> "UK issues bonds."
    """
    pred, args = _decompose_application(expr)

    pred = _prettify_token(pred)

    if len(args) == 1:
        subj = _prettify_token(str(args[0]))

        # subject phrase
        if _is_proper_noun(subj):
            subj_phrase = subj.upper() if subj in {"uk", "usa"} else subj.capitalize()
        else:
            subj_phrase = f"{_a_or_an(subj)} {subj}"

        # predicate phrase (noun vs adjective-ish)
        adjective_like = {
            "tradable", "diversified", "illegal",
            "risingprices", "fallingprices",
            "potentialloss"
        }
        if pred in adjective_like:
            return f"{subj_phrase.capitalize()} is {pred}."
        else:
            return f"{subj_phrase.capitalize()} is {_a_or_an(pred)} {pred}."

    if len(args) == 2:
        s = _prettify_token(str(args[0]))
        o = _prettify_token(str(args[1]))

        # small plural for nicer English on common objects
        plural_map = {"bond": "bonds", "stock": "stocks", "security": "securities"}
        o_phrase = plural_map.get(o, o)

        s_phrase = s.upper() if s in {"uk", "usa"} else s
        return f"{s_phrase} {pred} {o_phrase}."

    # fallback
    return str(expr)





# decomposing the text from tuple
def _decompose_application(expr) -> Tuple[str, List]:
    args = []
    while isinstance(expr, ApplicationExpression):
        args.append(expr.argument)
        expr = expr.function
    args.reverse()
    return str(expr), args

# functionality to unify the pattern of the tuple for explanation
def _unify(goal_expr, pattern_expr) -> Optional[Dict[str, object]]:
    gpred, gargs = _decompose_application(goal_expr)
    ppred, pargs = _decompose_application(pattern_expr)

    if gpred != ppred or len(gargs) != len(pargs):
        return None

    subs: Dict[str, object] = {}
    for p_arg, g_arg in zip(pargs, gargs):
        if isinstance(p_arg, IndividualVariableExpression):
            vname = p_arg.variable.name
            if vname in subs and str(subs[vname]) != str(g_arg):
                return None
            subs[vname] = g_arg
        else:
            if str(p_arg) != str(g_arg):
                return None
    return subs


def _apply_subs(expr, subs: Dict[str, object]):
    for vname, value_expr in subs.items():
        expr = expr.replace(Variable(vname), value_expr)
    return expr


def _is_fact(expr) -> bool:
    return not isinstance(expr, ImpExpression)


def _explain_goal(goal_expr, knowledge_base: List, visited: Optional[Set[str]] = None, max_depth: int = 8):
    if visited is None:
        visited = set()

    gstr = str(goal_expr)
    if gstr in visited or max_depth <= 0:
        return None
    visited.add(gstr)

    # direct fact
    for e in knowledge_base:
        if _is_fact(e) and str(e) == gstr:
            return [f"Fact: {gstr}"]

    # rules
    for rule in knowledge_base:
        if not isinstance(rule, ImpExpression):
            continue

        premise = rule.first
        conclusion = rule.second

        subs = _unify(goal_expr, conclusion)
        if subs is None:
            continue

        instantiated_premise = _apply_subs(premise, subs)
        chain = _explain_goal(instantiated_premise, knowledge_base, visited, max_depth - 1)
        if chain is not None:
            chain.append(f"Rule: {str(rule)}")
            chain.append(f"Therefore: {gstr}")
            return chain

    return None

# validating facts with explanation
def validate_fact_with_explanation(statement: str, knowledge_base: List, max_depth: int = 8) -> str:
    try:
        expr = read_expression(statement)

        if ResolutionProver().prove(expr, assumptions=knowledge_base):
            chain = _explain_goal(expr, knowledge_base, max_depth=max_depth)
            if chain:
                conclusion = _expr_to_english_sentence(expr)
                return "Correct.\nReason:\n- " + "\n- ".join(chain) + f"\nConclusion: {conclusion}"

        neg = read_expression(f"-({statement})")
        if ResolutionProver().prove(neg, assumptions=knowledge_base):
            return "Incorrect."

        return "I don’t know."

    except Exception as e:
        return f"Sorry, I couldn’t process that check ({e})."
