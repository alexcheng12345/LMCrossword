import re


def extract_required_length(prompt):
    for message in reversed(prompt):
        content = str(message.get("content", ""))
        matches = re.findall(r"\((\d+)\)//", content)
        if matches:
            return int(matches[-1])
    return None


def normalize_answer(answer):
    text = str(answer).strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[0] if lines else text
    candidate = re.sub(r"^(?:clue|answer)\s*:\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"^(?:my\s+answer\s+is|the\s+answer\s+is)\s*", "", candidate, flags=re.I)
    return "".join(character for character in candidate if character.isalpha())


def is_valid_answer(answer, required_length):
    if required_length is None:
        return True
    return len(normalize_answer(answer)) == required_length


def retry_message(required_length, invalid_answer):
    actual_length = len(normalize_answer(invalid_answer))
    return (
        f"Your previous answer was rejected because it has {actual_length} "
        f"alphabetic characters, but exactly {required_length} are required. "
        f"Reply with only the answer word. It must contain exactly "
        f"{required_length} alphabetic characters."
    )
