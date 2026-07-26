import math
import re
from fractions import Fraction

FINAL_TEMPLATE = "Final Answer: The final answer is {answer}. I hope it is correct."


def strip_boxed(text: str) -> str:
    text = text.strip()
    for prefix in (r"\boxed{", "boxed{"):
        start = text.rfind(prefix)
        if start < 0:
            continue
        index = start + len(prefix)
        depth = 1
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    tail = text[index + 1 :].strip()
                    if tail in {"", "."}:
                        return text[start + len(prefix) : index].strip()
                    break
            index += 1
    return text


def normalize_answer(answer: str) -> str:
    answer = strip_boxed(str(answer))
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\u2212", "-")
    answer = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.rstrip(".")
    return answer


def extract_math_answer(solution: str) -> str | None:
    boxed = strip_boxed(solution)
    if boxed != solution.strip():
        return normalize_answer(boxed)
    markers = ["Final Answer: The final answer is", "answer is", "Answer:", "answer:"]
    for marker in markers:
        index = solution.rfind(marker)
        if index >= 0:
            tail = solution[index + len(marker) :].split("I hope it is correct", 1)[0]
            return normalize_answer(tail.split("\n", 1)[0])
    return None


def extract_final_sentence_answer(completion: str) -> str | None:
    match = re.search(
        r"Final Answer:\s*The final answer is\s*(.+?)\.\s*I hope it is correct\.",
        completion,
        flags=re.DOTALL,
    )
    if match:
        return normalize_answer(match.group(1))
    return extract_math_answer(completion)


def _canonical_text(answer: str) -> str:
    normalized = normalize_answer(answer)
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("{", "").replace("}", "")
    return normalized.lower()


def _as_number(answer: str) -> float | None:
    text = _canonical_text(answer)
    text = text.replace(",", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    frac_match = re.fullmatch(r"(-?)\\frac(-?\d+(?:\.\d+)?)(-?\d+(?:\.\d+)?)", text)
    if not frac_match:
        frac_match = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", text)
    try:
        if frac_match:
            sign, numerator, denominator = frac_match.groups()
            value = float(numerator) / float(denominator)
            return -value if sign == "-" else value
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text)
        if re.fullmatch(r"-?\d+/\d+", text):
            return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return None


def answers_match(predicted: str | None, target: str) -> bool:
    if predicted is None:
        return False
    if _canonical_text(predicted) == _canonical_text(target):
        return True
    pred_num = _as_number(predicted)
    target_num = _as_number(target)
    return (
        pred_num is not None
        and target_num is not None
        and math.isclose(
            pred_num,
            target_num,
            rel_tol=1e-8,
            abs_tol=1e-8,
        )
    )


def build_worked_text(problem: str, reasoning: str, answer: str) -> str:
    reasoning = reasoning.strip()
    answer = normalize_answer(answer)
    if reasoning and not reasoning.endswith((".", "!", "?")):
        reasoning += "."
    return f"Problem:\n{problem.strip()}\n\nSolution:\n{reasoning}\n{FINAL_TEMPLATE.format(answer=answer)}"


def build_prompt(problem: str) -> str:
    return f"Problem:\n{problem.strip()}\n\nSolution:\n"
