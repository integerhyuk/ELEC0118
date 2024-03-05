import re

import torch.nn.functional as F
import numpy as np
import textgrid
import torch
from torch import nn

from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          WhisperFeatureExtractor,
                          WhisperTokenizer,
                          WhisperProcessor, BatchEncoding)

from src.config import WAV2VEC, WHISPER


def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""

    ind = 0
    str_interval = str(IntervalObject)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name

    return (pn, st, et)


def parse_textgrid(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList("words")
    words_list = list_words[0]

    result = []
    for ele in words_list:
        d = parse_Interval(ele)
        result.append(d)
    return result


def create_processor(model_name_or_path: object, vocab_file: object = None, type: object = WHISPER) -> object:
    if type == WAV2VEC:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path)
    else:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name_or_path)

    if vocab_file:
        if type == WAV2VEC:
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file,
                do_lower_case=False,
                word_delimiter_token="|",
            )
        else:
            tokenizer = WhisperTokenizer(
                vocab_file,
                do_lower_case=False,
            )
    else:
        if type == WAV2VEC:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name_or_path,
                do_lower_case=False,
                word_delimiter_token="|",
            )
        else:
            tokenizer = WhisperTokenizer.from_pretrained(
                model_name_or_path,
                do_lower_case=False,
            )
    if type == WAV2VEC:
        return Wav2Vec2Processor(feature_extractor, tokenizer)
    else:
        return WhisperProcessor(feature_extractor, tokenizer)


def prepare_example(text, vocabulary_text_cleaner):
    # Normalize and clean up text; order matters!
    try:
        text = " ".join(text.split())  # clean up whitespaces
    except:
        text = "NULL"
    updated_text = text
    # updated_text = vocabulary_text_cleaner.sub("", updated_text)
    if updated_text != text:
        return re.sub(' +', ' ', updated_text).strip()
    else:
        return re.sub(' +', ' ', text).strip()


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +", ' ', text).strip()

    return text


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    for key in checkpoint['state_dict'].copy():
        if not 'transformer' in key:
            del checkpoint['state_dict'][key]
        else:
            new_key = key.lstrip("transformer").lstrip(".")
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    x, y = model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(x)
    print(y)

    return model


class Classic_Attention(nn.Module):
    def __init__(self, input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim, embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights, 1)
        return attention_weights_normalized


def downsample(x, x_len, sample_rate=2):
    batch_size, timestep, feature_dim = x.shape
    x_len = x_len // sample_rate
    # Drop the redundant frames and concat the rest according to sample rate
    if timestep % sample_rate != 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    if timestep % sample_rate != 0:
        x = x.contiguous().view(batch_size, int(timestep // sample_rate) + 1, feature_dim * sample_rate)
    else:
        x = x.contiguous().view(batch_size, int(timestep // sample_rate), feature_dim * sample_rate)

    return x


def create_mask(batch_size, seq_len, spec_len):
    with torch.no_grad():
        attn_mask = torch.ones((batch_size, seq_len))  # (batch_size, seq_len)

        for idx in range(batch_size):
            # zero vectors for padding dimension
            attn_mask[idx, spec_len[idx]:] = 0

    return attn_mask


def label2idx(label):
    label2idx = {
        "hap": 0,
        "ang": 1,
        "neu": 2,
        "sad": 3,
        "exc": 0}

    return label2idx[label]


def evaluate_metrics(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)
    ua = np.mean(pred_label.astype(int) == true_label.astype(int))
    pred_onehot = np.eye(4)[pred_label.astype(int)]
    true_onehot = np.eye(4)[true_label.astype(int)]
    wa = np.mean(np.sum((pred_onehot == true_onehot) * true_onehot, axis=0) / np.sum(true_onehot, axis=0))
    key_metric, report_metric = 0.9 * wa + 0.1 * ua, {'wa': wa, 'ua': ua}
    return wa, report_metric  # take wa as key metric


# TODO: Implement the following functions
def audio_process(wav: np.ndarray) -> BatchEncoding:
    """

    :param wav: waveform

    1. split wav into chunks
    2. Using Wav2Vec2 tokenizer, process each chunk and return BatchEncoding('wav_input')
    3. Using Wisper model, process each chunk and return input_text
    4. Using Bert tokenizer, process input_text and return BatchEncoding('text_input')
    5. return BatchEncoding{'wav_input': BatchEncoding,
                            'text_input': BatchEncoding}

    :return: BatchEncoding
    """

    def text_process(wav: str) -> BatchEncoding:
        pass

    return BatchEncoding