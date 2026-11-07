"""The vote key for problems answered in \\boxed{} (ArXivMath), the only change to Part 4's vote in this part.

Part 4's runner (code/strategies/common.py, answer_key) decides which branches give the same answer. For a problem whose
answer is the last \\boxed{...} of the reply, this part added one case at the top of answer_key, after `final` is set:

    if task.get('subcategory') == 'arxivmath':          # the answer is the last \\boxed{}
        box = last_boxed(final)
        return None if box is None else ('box', box_key(box))

with the two functions below. Two boxed answers vote together when they are equal after removing spacing and
layout-only commands; other equivalent spellings (1/2 against 0.5, reordered sets) do not merge.

    python3 code/boxed_vote.py      runs the self-check
"""


def last_boxed(text):
    """The content of the last \\boxed{...} in text (balanced braces), or None."""
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    depth, j = 0, i + len('\\boxed')
    for k in range(j, len(text)):
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def box_key(box):
    """A boxed answer without spacing and layout-only commands (other equivalent spellings do not merge)."""
    for a, b in (('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'), ('\\left', ''), ('\\right', ''), ('\\displaystyle', ''),
                 ('\\,', ''), ('\\!', ''), ('\\;', ''), ('\\ ', '')):
        box = box.replace(a, b)
    return ''.join(box.split()).rstrip('.')


if __name__ == '__main__':
    assert last_boxed('first \\boxed{1}, then \\boxed{\\frac{1}{2}}.') == '\\frac{1}{2}'
    assert last_boxed('no answer') is None and last_boxed('\\boxed{unclosed') is None
    assert box_key('\\dfrac{1}{2}') == box_key('\\frac {1}{2}') == '\\frac{1}{2}'
    assert box_key('\\left( 2,\\, 3 \\right)') == '(2,3)'
    assert box_key('\\{1, 2\\}') != box_key('\\{2, 1\\}')          # reordered sets stay separate votes
    assert box_key('0.5') != box_key('\\frac{1}{2}')
    print('boxed_vote self-check passed')
