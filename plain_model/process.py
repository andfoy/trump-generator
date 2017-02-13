#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import progressbar

vec_repr = {}

DOT = '.'
COMMA = ','
COLON = ':'
SEMICOLON = ';'
QUOTATION = '"'
DOLLAR = '$'
PERCENT = '%'
EXCLAMATION = '!'
QUESTION = '?'
DIV = '/'
R_OPEN = '('
R_CLOSE = ')'
S_OPEN = '['
S_CLOSE = ']'
ELLIPSIS = '…'
LONG_DASH = '–'
IMP_ELLIP = '...'
EOL = '\n'

START = [IMP_ELLIP, QUOTATION, R_OPEN, S_OPEN, DOLLAR, ELLIPSIS]
END = [R_CLOSE, S_CLOSE, PERCENT, DOT, COMMA, COLON, SEMICOLON,
       EXCLAMATION, QUOTATION, QUESTION, ELLIPSIS, IMP_ELLIP]

ALL_SYM = [DOT, COMMA, COLON, SEMICOLON, QUOTATION, DOLLAR, PERCENT,
           EXCLAMATION, QUESTION, DIV, R_OPEN, R_CLOSE, S_OPEN,
           S_CLOSE, ELLIPSIS, LONG_DASH, IMP_ELLIP, EOL]

vec_repr = dict(zip(ALL_SYM, range(0, len(ALL_SYM))))


def process_word(word):
    l_accum = []
    r_accum = []
    # print len(word)
    while word[0] in START or word[:3] in START:
        for sym in START:
            if word.startswith(sym):
                l_accum.append(vec_repr[sym])
                if sym != IMP_ELLIP:
                    word = word[1:]
                else:
                    word = word[3:]
        if len(word) == 0:
            break
    if len(word) > 0:
        while word[-1] in END or word[-3:] in END:
            for sym in END:
                if word.endswith(sym):
                    r_accum = [vec_repr[sym]] + r_accum
                    if sym != IMP_ELLIP:
                        word = word[:-1]
                    else:
                        word = word[:-3]
            if len(word) == 0:
                break
    return word, l_accum, r_accum


def add_word(word):
    idx = len(vec_repr) - 1
    word = word.lower()
    if word not in vec_repr:
        vec_repr[word] = idx + 1
    return vec_repr[word]


def process_line(line):
    result = []
    if len(line) > 0:
        words = line.split(' ')
        for word in words:
            if len(word) > 0:
                # try:
                word, l_accum, r_accum = process_word(word)
                # except Exception as e:
                    # print word
                    # print line
                    # raise e
                if DIV in word:
                    spl_words = word.split(DIV)
                    for part in spl_words:
                        val = add_word(part)
                        l_accum.append(val)
                else:
                    val = add_word(word)
                    l_accum.append(val)
                result += l_accum + r_accum
    else:
        result.append(vec_repr[EOL])
    return result


def main():
    bar = progressbar.ProgressBar()
    _input = []
    with open('speeches.txt', 'rb') as fp:
        for line in bar(fp):
            line = line.rstrip()
            post_pr = process_line(line)
            _input += post_pr
        np.savez('dictionary', data=_input, vec_repr=vec_repr)


if __name__ == '__main__':
    main()
