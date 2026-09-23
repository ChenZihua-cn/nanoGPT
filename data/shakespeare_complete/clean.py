"""Clean a raw Project Gutenberg file into a nanoGPT-ready corpus.

Usage:
    python clean.py                 # reads pg100_raw.txt -> writes input.txt
    python clean.py pg123_raw.txt   # optional explicit input

What it does:
    1. cut everything outside the *** START / END OF THE PROJECT GUTENBERG *** markers
    2. cut the front matter (title page + the repeated table of contents)
    3. strip trailing per-line line numbers that Gutenberg appends to verse
    4. fold curly quotes / em dashes / accented letters down to ASCII
       so the vocabulary stays close to the original 65-char tiny shakespeare set
    5. collapse runs of 3+ blank lines into a single blank line

Then run:  python prepare.py     (generates train.bin / val.bin / meta.pkl)
"""
import os
import re
import sys
import unicodedata

START = '*** START OF THE PROJECT GUTENBERG EBOOK'
END = '*** END OF THE PROJECT GUTENBERG EBOOK'

# verse lines end with two-or-more spaces then a 1-4 digit line number
LINE_NUMBER = re.compile(r'^(.*\S)\s{2,}\d{1,4}\s*$')

TRANSLATION = {
    '\u2019': "'", '\u2018': "'", '\u02bc': "'", '\u2032': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u00ab': '"', '\u00bb': '"',
    '\u2014': '-', '\u2013': '-', '\u2012': '-', '\u2044': '/',
    '\u2026': '...', '\u00a0': ' ', '\u2020': '*', '\u2021': '*',
}


def to_ascii(text):
    for src, dst in TRANSLATION.items():
        text = text.replace(src, dst)
    # anything still non-ascii (æ, é, œ, ...) gets decomposed and the accents dropped
    text = unicodedata.normalize('NFKD', text)
    return text.encode('ascii', 'ignore').decode('ascii')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'pg100_raw.txt')
    out_path = os.path.join(here, 'input.txt')

    raw = open(raw_path, 'rb').read().decode('utf-8-sig')
    lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    start_idx = next(i for i, l in enumerate(lines) if START in l)
    end_idx = next(i for i, l in enumerate(lines) if END in l)
    body = lines[start_idx + 1:end_idx]

    # drop the front matter: keep from the first real content heading onwards
    heads = [i for i, l in enumerate(body[:400]) if l.strip() == 'THE SONNETS']
    if len(heads) >= 2:
        body = body[heads[1]:]

    # ragged trailing whitespace + Gutenberg verse line numbers
    stripped_nums = 0
    cleaned = []
    for line in body:
        line = line.rstrip()
        m = LINE_NUMBER.match(line)
        if m:
            line = m.group(1)
            stripped_nums += 1
        cleaned.append(line)

    text = '\n'.join(cleaned).strip() + '\n'
    text = to_ascii(text)
    # this edition marks italics with underscores: [_Exit Countess._] -> [Exit Countess.]
    text = text.replace('_', '').replace('\t', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    vocab = sorted(set(text))
    print(f'in  : {raw_path}   ({len(raw):,} bytes)')
    print(f'out : {out_path}')
    print(f'characters      : {len(text):,}')
    print(f'unique chars    : {len(vocab)}')
    print(f'  {"".join(c if c != chr(10) else chr(92) + "n" for c in vocab)}')
    print(f'line numbers stripped: {stripped_nums:,}')


if __name__ == '__main__':
    main()
