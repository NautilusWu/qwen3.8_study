"""Streaming loop detector for a reasoning reply. Feed it the reply text as it grows; it says when the reply is looping.

Two rules (chosen offline on PLAN_07's replies, where neither flagged an answered reply):
  exact   the last R characters repeat with a period p <= PMAX (text[-(R+p):-p] == text[-R:]); checked every CHECK
          characters. The cut point is where that periodic stretch begins, so the loop itself can be dropped.
  phrase  among the last WINDOW words, more than SHARE of the word N-grams already occurred earlier in the reply,
          at STREAK consecutive checks (every PHRASE_EVERY words, once the reply has WINDOW + MIN_BEFORE words).
          Cut point: the current end. (Without the streak it also fired on legitimate repetitive work, such as
          writing out the rows of a cellular automaton.)
"""
import re

R, PMAX, CHECK = 2000, 2000, 512
N, WINDOW, SHARE, MIN_BEFORE, PHRASE_EVERY, STREAK = 8, 3000, 0.5, 1000, 500, 4
WORD = re.compile(r'\S+')


class LoopDetector:
    def __init__(self):
        self.text = ''
        self.next_exact = R + 2
        self.next_phrase = WINDOW + MIN_BEFORE
        self.streak = 0

    def feed(self, text):
        """The whole reply so far; returns None or {'rule', 'cut', 'at'} (character offsets into the text)."""
        self.text = text
        while len(text) >= self.next_exact:
            pos = self.next_exact
            self.next_exact += CHECK
            hit = self._exact(pos)
            if hit:
                return hit
        hit = self._phrase()
        return hit

    def _exact(self, pos):
        w = self.text[:pos]
        for p in range(1, PMAX + 1):
            if len(w) < R + p:
                break
            if w[-p:] == w[-2 * p:-p] and w[-(R + p):-p] == w[-R:]:
                start = len(w) - R - p
                while start > 0 and w[start - 1] == w[start - 1 + p]:
                    start -= 1
                return {'rule': 'exact', 'period': p, 'cut': start, 'at': pos}
        return None

    def _phrase(self):
        words = WORD.findall(self.text)
        if len(words) < self.next_phrase:
            return None
        while len(words) >= self.next_phrase:
            self.next_phrase += PHRASE_EVERY
        first = {}                      # rebuilt each check: the last word of a stream may still be growing
        for i in range(len(words) - N + 1):
            first.setdefault(tuple(words[i:i + N]), i)
        start = len(words) - WINDOW
        tail = range(start, len(words) - N + 1)
        share = sum(first[tuple(words[i:i + N])] < start for i in tail) / len(tail)
        self.streak = self.streak + 1 if share > SHARE else 0
        if self.streak >= STREAK:
            return {'rule': 'phrase', 'share': round(share, 3), 'cut': len(self.text), 'at': len(self.text)}
        return None
