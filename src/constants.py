_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)

