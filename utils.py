# --*-- coding:utf-8 --*--
# @author: xiao shanghua

import re


def segment(text):
    text = re.sub(r'[,.!?;"\'\-_+\[\]()]', '', text.lower())
    segs = re.split(r'\s', text)
    return segs



