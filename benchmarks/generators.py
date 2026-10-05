"""Task builders. Real datasets (ARC, GSM8K) are loaded from ``data/raw``; synthetic tasks use seeded,
solver-verified generators so ground truth never comes from an LLM."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from harness.core.types import Example

RAW = Path("data/raw")
TASKS = ["factual_qa", "math", "logic", "longform", "instruction", "code"]
LETTERS = "ABCD"


# ----------------------------------------------------------------------------- factual QA (ARC)
def build_factual_qa(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    items: list[tuple[str, dict]] = []
    for ds, path in (("ARC-Challenge", RAW / "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"),
                     ("ARC-Easy", RAW / "ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl")):
        for line in path.read_text().splitlines():
            d = json.loads(line)
            ch = d["question"]["choices"]
            if len(ch) == 4 and d["answerKey"] in "ABCD1234":
                items.append((ds, d))
    rng.shuffle(items)
    out = []
    for ds, d in items[:n]:
        ch = d["question"]["choices"]
        key = "ABCD"["1234".index(d["answerKey"])] if d["answerKey"] in "1234" else d["answerKey"]
        opts = "\n".join(f"{LETTERS[i]}. {c['text']}" for i, c in enumerate(ch))
        stem = d["question"]["stem"]
        prompt = (f"Answer the following multiple-choice science question. Explain your reasoning briefly, "
                  f"then finish with 'The answer is X' where X is A, B, C or D.\n\nQuestion: {stem}\n{opts}\n\n"
                  f"Let's think step by step.")
        out.append(Example(f"factual_qa/{d['id']}", "factual_qa", prompt, key,
                           {"question": stem, "dataset": ds, "difficulty": 2 if ds == "ARC-Challenge" else 1}))
    return out


# ----------------------------------------------------------------------------- math (GSM8K)
def _synthetic_arith(n: int, seed: int) -> list[Example]:
    """1-3 step integer word problems; ground truth computed directly."""
    rng = random.Random(seed + 101)
    items = ["apples", "marbles", "stickers", "cookies", "books", "coins"]
    out = []
    for i in range(n):
        steps = rng.choice([1, 1, 2])
        name, item = rng.choice(_NAMES), rng.choice(items)
        val = rng.randint(3, 15)
        text = f"{name} has {val} {item}."
        for _ in range(steps):
            op = rng.choice(["+", "+", "-", "*"])
            if op == "+":
                k = rng.randint(2, 9); val += k; text += f" {name} gets {k} more {item}."
            elif op == "-":
                k = rng.randint(1, max(1, min(val - 1, 9))); val -= k; text += f" {name} gives away {k} {item}."
            else:
                k = rng.randint(2, 4); val *= k; text += f" Then the number of {item} {name} has is multiplied by {k}."
        q = f"{text} How many {item} does {name} have now?"
        prompt = (f"Solve the math problem. Think step by step and finish with 'The answer is N' "
                  f"where N is a number.\n\nQ: {q}\n\nLet's think step by step.")
        out.append(Example(f"math/arith-{seed}-{i}", "math", prompt, str(val),
                           {"question": q, "dataset": "synthetic-arith", "difficulty": steps}))
    return out


def build_math(n: int, seed: int) -> list[Example]:
    """Half GSM8K (hard, real), half synthetic arithmetic (graded difficulty)."""
    return _build_gsm8k(n // 2, seed) + _synthetic_arith(n - n // 2, seed)


def _build_gsm8k(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (RAW / "gsm8k_test.jsonl").read_text().splitlines()]
    rng.shuffle(rows)
    out = []
    for i, d in enumerate(rows[:n]):
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        n_steps = d["answer"].count("\n")  # number of reasoning lines in the reference solution
        prompt = (f"Solve the math problem. Think step by step and finish with 'The answer is N' "
                  f"where N is a number.\n\nQ: {d['question']}\n\nLet's think step by step.")
        out.append(Example(f"math/gsm8k-{seed}-{i}", "math", prompt, gold,
                           {"question": d["question"], "dataset": "GSM8K", "difficulty": n_steps}))
    return out


# ----------------------------------------------------------------------------- logic (synthetic syllogisms)
_KINDS = ["zorbs", "blicks", "wugs", "fleems", "norgs", "plinks", "drabs", "quilts", "snerds", "tovs", "vexes", "glorps"]
_NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Niaj"]


def _singular(k: str) -> str:
    return k[:-1]


def build_logic(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        hops = rng.choice([1, 2, 3])
        label = ["yes", "no", "unknown"][i % 3]
        kinds = rng.sample(_KINDS, hops + 2)
        who = rng.choice(_NAMES)
        chain = [f"All {kinds[j]} are {kinds[j + 1]}." for j in range(hops)]
        facts = [f"{who} is a {_singular(kinds[0])}."]
        if label == "yes":
            q_kind = kinds[hops]
        elif label == "no":
            chain.append(f"No {kinds[hops]} are {kinds[hops + 1]}.")
            q_kind = kinds[hops + 1]
        else:
            q_kind = kinds[hops + 1]  # unrelated to the chain
            if hops >= 1:
                chain = chain[:hops]
        premises = chain + facts
        rng.shuffle(premises)
        question = f"Is {who} a {_singular(q_kind)}?"
        prompt = ("Read the premises and decide whether the conclusion must be true. Explain briefly, then finish "
                  "with 'The answer is yes', 'The answer is no' or 'The answer is unknown' (use unknown if it "
                  "cannot be determined).\n\nPremises: " + " ".join(premises) + f"\nQuestion: {question}\n\n"
                  "Let's think step by step.")
        out.append(Example(f"logic/{seed}-{i}", "logic", prompt, label,
                           {"question": question, "dataset": "synthetic-syllogism", "difficulty": hops}))
    return out


# ----------------------------------------------------------------------------- long-form reasoning (orderings)
_REL = [("taller", "tallest", "shortest"), ("older", "oldest", "youngest"), ("faster", "fastest", "slowest")]


def build_longform(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([3, 4, 5, 6])
        names = rng.sample(_NAMES, k)  # names[0] is greatest
        comp, sup, inf = rng.choice(_REL)
        stmts = [(names[j], names[j + 1]) for j in range(k - 1)]
        rng.shuffle(stmts)
        text = " ".join(f"{a} is {comp} than {b}." for a, b in stmts)
        ask_top = rng.random() < 0.5
        gold = names[0] if ask_top else names[-1]
        q = f"Who is the {sup if ask_top else inf}?"
        prompt = (f"Read the statements and answer the question. Explain your reasoning step by step, "
                  f"then finish with 'The answer is NAME'.\n\nStatements: {text}\nQuestion: {q}\n\n"
                  f"Let's think step by step.")
        out.append(Example(f"longform/{seed}-{i}", "longform", prompt, gold,
                           {"question": q, "dataset": "synthetic-ordering", "difficulty": k,
                            "order": names, "comparative": comp}))
    return out


# ----------------------------------------------------------------------------- instruction following (synthetic IFEval-style)
_TOPICS = ["the ocean", "a rainy day", "learning to cook", "the moon", "a small village", "teamwork",
           "recycling", "a long train ride", "bees", "winter mornings", "a library", "friendship"]
_KEYWORDS = ["river", "light", "quiet", "bridge", "garden", "story", "window", "energy", "market", "forest"]


def _constraint(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["min_words", "max_words", "keywords", "lowercase", "no_commas", "end_phrase",
                       "n_sentences", "quotes"])
    if kind == "min_words":
        return {"kind": kind, "n": rng.choice([40, 60, 80]), "text": None}
    if kind == "max_words":
        return {"kind": kind, "n": rng.choice([15, 25, 35])}
    if kind == "keywords":
        return {"kind": kind, "words": rng.sample(_KEYWORDS, 2)}
    if kind == "end_phrase":
        return {"kind": kind, "phrase": rng.choice(["That is all.", "Thank you for reading."])}
    if kind == "n_sentences":
        return {"kind": kind, "n": rng.choice([2, 3, 4])}
    return {"kind": kind}


def constraint_text(c: dict[str, Any]) -> str:
    k = c["kind"]
    return {
        "min_words": lambda: f"Use at least {c['n']} words.",
        "max_words": lambda: f"Use fewer than {c['n']} words.",
        "keywords": lambda: f"Include the words '{c['words'][0]}' and '{c['words'][1]}'.",
        "lowercase": lambda: "Write your entire response in lowercase letters.",
        "no_commas": lambda: "Do not use any commas.",
        "end_phrase": lambda: f"End your response with the exact phrase '{c['phrase']}'.",
        "n_sentences": lambda: f"Write exactly {c['n']} sentences.",
        "quotes": lambda: "Wrap your entire response in double quotation marks.",
    }[k]()


def build_instruction(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        n_c = rng.choice([1, 2, 3])
        cons: list[dict] = []
        while len(cons) < n_c:
            c = _constraint(rng)
            if c["kind"] not in {x["kind"] for x in cons} and not (
                    c["kind"] in ("min_words", "max_words") and {"min_words", "max_words"} & {x["kind"] for x in cons}):
                cons.append(c)
        topic = rng.choice(_TOPICS)
        prompt = f"Write a short paragraph about {topic}. " + " ".join(constraint_text(c) for c in cons)
        out.append(Example(f"instruction/{seed}-{i}", "instruction", prompt, cons,
                           {"question": f"Write a short paragraph about {topic}.", "dataset": "synthetic-ifeval-style",
                            "difficulty": n_c}))
    return out


# ----------------------------------------------------------------------------- code (one-line lambdas, executable tests)
def build_code(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)

    def tmpl() -> tuple[str, list[tuple[Any, Any]], int]:
        a, b = rng.randint(2, 9), rng.randint(1, 9)
        t = rng.randrange(19)
        if t == 0:
            return f"returns x multiplied by {a} plus {b}", [(x, x * a + b) for x in (0, 1, 5, -3)], 1
        if t == 1:
            return "returns the sum of a list of numbers x", [(v, sum(v)) for v in ([1, 2, 3], [], [10, -4])], 1
        if t == 2:
            return "returns the string s reversed", [(v, v[::-1]) for v in ("abc", "", "hello")], 1
        if t == 3:
            return "returns the length of the string s", [(v, len(v)) for v in ("abc", "", "hello!")], 1
        if t == 4:
            return "returns the largest number in the list x", [(v, max(v)) for v in ([1, 5, 2], [-3, -1], [7])], 1
        if t == 5:
            return f"returns True if x is divisible by {a}, otherwise False", [(x, x % a == 0) for x in range(0, 20, 3)], 2
        if t == 6:
            return "returns the string s in uppercase letters", [(v, v.upper()) for v in ("abc", "Hello", "")], 1
        if t == 7:
            return "returns the list x sorted in ascending order", [(v, sorted(v)) for v in ([3, 1, 2], [], [5, 5, 1])], 1
        if t == 8:
            return f"returns the square of x plus {b}", [(x, x * x + b) for x in (0, 2, -3, 5)], 2
        if t == 9:
            return "returns the first element of the list x", [(v, v[0]) for v in ([4, 5], ["a"], [9, 8, 7])], 1
        if t == 10:
            return "returns the number of even numbers in the list x", [(v, len([y for y in v if y % 2 == 0])) for v in ([1, 2, 4], [], [3, 5], [2])], 3
        if t == 11:
            return f"returns the sum of x and its double, minus {b}", [(x, x + 2 * x - b) for x in (0, 1, 4, -2)], 3
        if t == 12:
            return "returns the string s repeated three times", [(v, v * 3) for v in ("ab", "", "x")], 2
        if t == 13:
            return "returns True if the string s reads the same forwards and backwards", [(v, v == v[::-1]) for v in ("aba", "abc", "", "xyyx")], 3
        if t in (14, 15):
            return f"returns x plus {a}", [(x, x + a) for x in (0, 1, 7, -4)], 0
        if t == 16:
            return f"returns x multiplied by {a}", [(x, x * a) for x in (0, 1, 7, -4)], 0
        if t == 17:
            return f"returns x minus {b}", [(x, x - b) for x in (0, 1, 7, -4)], 0
        return f"returns the string s followed by the letter {chr(96 + a)}", [(v, v + chr(96 + a)) for v in ("ab", "", "x")], 0

    out = []
    for i in range(n):
        desc, tests, diff = tmpl()
        prompt = ("Write a Python lambda expression for each task. Every answer starts with the word lambda.\n\n"
                  "Task: returns x plus 1\nAnswer: lambda x: x + 1\n\n"
                  "Task: returns the string s followed by an exclamation mark\nAnswer: lambda s: s + '!'\n\n"
                  "Task: returns the list x without its first element\nAnswer: lambda x: x[1:]\n\n"
                  f"Task: {desc}\nAnswer:")
        out.append(Example(f"code/{seed}-{i}", "code", prompt, [[json.dumps(a), json.dumps(b)] for a, b in tests],
                           {"question": f"Write a Python lambda expression that {desc}.", "dataset": "synthetic-lambda",
                            "difficulty": diff}))
    return out


BUILDERS = {"factual_qa": build_factual_qa, "math": build_math, "logic": build_logic,
            "longform": build_longform, "instruction": build_instruction, "code": build_code}


def build_all(n_per_task: dict[str, int], seed: int) -> list[Example]:
    out: list[Example] = []
    for t, n in n_per_task.items():
        out.extend(BUILDERS[t](n, seed))
    return out
