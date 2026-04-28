"""@NamBert"""

import torch
import pypinyin
import numpy as np
from pathlib import Path
from PIL import ImageFont


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def is_chinese(uchar):
    return '\u4e00' <= uchar <= '\u9fa5'


def is_float(string):
    try:
        float(string)
        return True
    except:
        return False


def convert_char_to_image(character, font_size=32):
    font = ImageFont.truetype("src/ms_yahei.ttf", size=font_size)

    image = font.getmask(character)
    image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])

    image = image[:font_size, :font_size]

    if image.size != (font_size, font_size):
        back_image = np.zeros((font_size, font_size)).astype(np.float32)
        offset0 = (font_size - image.shape[0]) // 2
        offset1 = (font_size - image.shape[1]) // 2
        back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
        image = back_image

    return torch.tensor(image)


def convert_char_to_pinyin(character, size=-1, tone=False):
    if not is_chinese(character):
        return torch.LongTensor([0] * max(size, 1))

    if tone:
        pinyin = pypinyin.pinyin(character, style=pypinyin.TONE3)[0][0]
    else:
        pinyin = pypinyin.pinyin(character, style=pypinyin.NORMAL)[0][0]

    if not tone:
        embeddings = torch.tensor([ord(letter) - 96 for letter in pinyin])
    else:
        embeddings = []
        for letter in pinyin:
            if letter.isnumeric():
                embeddings.append(int(letter) + 27)
            else:
                embeddings.append(ord(letter) - 96)
        embeddings = torch.tensor(embeddings)

    if size > len(embeddings):
        padding = torch.zeros(size - len(embeddings))
        embeddings = torch.concat([embeddings, padding])

    return embeddings


def pred_token_process(src_tokens, pred_tokens, ignore_token: list = None):
    if len(src_tokens) != len(pred_tokens):
        print("[Error]unequal length:", ''.join(src_tokens))
        return pred_tokens

    for i in range(len(src_tokens)):
        if len(src_tokens[i]) != len(pred_tokens[i]):
            # print("[Warning]unequal token length: %s, token: (%s, %s)"
            #       % (''.join(src_tokens), src_tokens[i], pred_tokens[i]))
            pred_tokens[i] = src_tokens[i]
            continue

        if not is_chinese(src_tokens[i]):
            pred_tokens[i] = src_tokens[i]
            continue

        if ignore_token:
            if src_tokens[i] in ignore_token:
                pred_tokens[i] = src_tokens[i]
                continue

    return pred_tokens