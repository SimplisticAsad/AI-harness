from benchmarks.generators import build_all
from benchmarks.graders import check_constraint, grade, ordering_violations, run_code_tests
from harness.signals.answers import extract_letter, extract_number, extract_yn


def test_extractors():
    assert extract_letter("so the answer is C.") == "C"
    assert extract_number("5 + 3 = 8. The answer is 1,200.") == "1200"
    assert extract_yn("It follows. The answer is no") == "no"


def test_gold_answers_grade_correct_for_each_task():
    ex = {e.task: e for e in build_all({"factual_qa": 1, "math": 2, "logic": 1, "longform": 1}, 3)}
    assert grade(ex["factual_qa"], f"The answer is {ex['factual_qa'].reference}").correct
    assert grade(ex["logic"], f"The answer is {ex['logic'].reference}").correct
    assert grade(ex["longform"], f"The answer is {ex['longform'].reference}").correct
    assert grade(ex["math"], f"The answer is {ex['math'].reference}").correct


def test_wrong_and_unparseable_are_errors_with_taxonomy():
    e = build_all({"logic": 1}, 3)[0]
    wrong = "no" if e.reference != "no" else "yes"
    g = grade(e, f"The answer is {wrong}")
    assert not g.correct and g.error_type == "logical_error"
    assert grade(e, "I like turtles").error_type == "incomplete_answer"


def test_longform_requires_valid_derivation():
    e = build_all({"longform": 1}, 3)[0]
    order, comp = e.meta["order"], e.meta["comparative"]
    bad = f"{order[-1]} is {comp} than {order[0]}. The answer is {e.reference}."
    assert ordering_violations(e, bad) == 1
    g = grade(e, bad)
    assert not g.correct and g.error_type == "reasoning_error"


def test_instruction_constraints():
    assert check_constraint({"kind": "lowercase"}, "all lower")
    assert not check_constraint({"kind": "lowercase"}, "Not lower")
    assert check_constraint({"kind": "n_sentences", "n": 2}, "One. Two.")
    assert check_constraint({"kind": "end_phrase", "phrase": "That is all."}, "Hi. That is all.")
    assert not check_constraint({"kind": "no_commas"}, "a, b")


def test_code_execution_sandbox():
    tests = [["1", "2"], ["5", "6"]]
    assert run_code_tests("lambda x: x + 1", tests) == "pass"
    assert run_code_tests("lambda x: x + 2", tests) == "wrong"
    assert run_code_tests("lambda x: (", tests) == "syntax"
    assert run_code_tests("lambda x: __import__('os')", tests) in ("runtime", "wrong")  # builtins are restricted
