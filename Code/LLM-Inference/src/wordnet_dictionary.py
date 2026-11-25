import re
from collections import defaultdict
from pathlib import Path


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize_word(value):
    return "".join(char for char in str(value).lower() if char.isalpha())


def tokenize(value):
    return {
        token
        for token in re.findall(r"[a-z]+", str(value).lower())
        if token not in STOP_WORDS
    }


class WordNetDictionary:
    def __init__(self, required_lengths, data_dir=None):
        try:
            import nltk
            from nltk.corpus import wordnet
        except ImportError as exc:
            raise RuntimeError(
                "Install NLTK with `pip install -r ..\\requirements.txt` "
                "before running B3."
            ) from exc

        if data_dir is not None:
            dictionary_path = Path(data_dir).resolve()
            if not dictionary_path.is_dir():
                raise NotADirectoryError(
                    f"WordNet data directory does not exist: {dictionary_path}"
                )
            nltk.data.path.insert(0, str(dictionary_path))

        lengths = set(required_lengths)
        entries = defaultdict(lambda: {"definitions": set()})

        try:
            for synset in wordnet.all_synsets():
                definition = synset.definition()
                examples = synset.examples()
                description = " ".join([definition, *examples]).strip()
                for lemma in synset.lemma_names():
                    word = normalize_word(lemma)
                    if len(word) not in lengths:
                        continue
                    if description:
                        entries[word]["definitions"].add(description)
        except LookupError as exc:
            raise RuntimeError(
                "The local NLTK WordNet corpus is not installed. Run "
                "`python -m nltk.downloader wordnet` with the project "
                "environment, then retry B3."
            ) from exc

        self._wordnet = wordnet
        self._entries = {
            word: {
                "word": word,
                "definitions": sorted(values["definitions"]),
            }
            for word, values in entries.items()
        }

    def retrieve_candidates(self, clue, search_terms, required_length, limit=30):
        query_tokens = tokenize(clue)
        for term in search_terms:
            query_tokens.update(tokenize(term))

        normalized_terms = {
            normalize_word(term)
            for term in search_terms
            if normalize_word(term)
        }
        synonym_scores = defaultdict(int)
        for term in search_terms:
            lookup_term = str(term).strip().lower().replace(" ", "_")
            for synset in self._wordnet.synsets(lookup_term):
                for lemma in synset.lemma_names():
                    word = normalize_word(lemma)
                    if len(word) == required_length:
                        synonym_scores[word] += 1

        ranked = []
        for word, entry in self._entries.items():
            if len(word) != required_length:
                continue

            definitions = entry["definitions"]
            candidate_tokens = tokenize(word)
            for definition in definitions:
                candidate_tokens.update(tokenize(definition))

            overlap = len(query_tokens & candidate_tokens)
            exact_term_score = 5.0 if word in normalized_terms else 0.0
            synonym_score = min(synonym_scores[word], 3) * 2.0
            overlap_score = overlap / max(1, len(query_tokens))
            score = exact_term_score + synonym_score + overlap_score
            if score <= 0:
                continue

            ranked.append(
                {
                    "word": word,
                    "definitions": definitions[:2],
                    "score": round(score, 4),
                }
            )

        ranked.sort(key=lambda candidate: (-candidate["score"], candidate["word"]))
        return ranked[:limit]
