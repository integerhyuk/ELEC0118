import os

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.const import vocabulary_text_cleaner, audio_processor, tokenizer
from src.utils import text_preprocessing, prepare_example, label2idx


class IEMOCAPDataset(Dataset):
    def __init__(self, config, data_list):
        self.data_list = data_list
        # self.unit_length = int(8 * 16000)
        self.audio_length = config['acoustic']['audio_length']
        self.feature_name = config['acoustic']['feature_name']
        self.feature_dim = config['acoustic']['embedding_dim']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, bert_text, label, asr_text, _  = self.data_list[index]
        audio_name = os.path.basename(audio_path)

        #------------- extract the audio features -------------#
        wave,sr = librosa.core.load(audio_path + ".wav", sr=None)
        if len(wave)>210000:
            wave = wave[:210000]
        audio_length = len(wave)

        bert_text = text_preprocessing(bert_text)

        #------------- clean asr target text -------------#
        asr_text = prepare_example(asr_text,vocabulary_text_cleaner)

        #------------- labels -------------#
        label = label2idx(label)

        #------------- wrap up all the output info the dict format -------------#
        return {'audio_input':wave,'text_input':bert_text,'audio_length':audio_length,
                'label':label,'audio_name':audio_name,'asr_target':asr_text,
                }


def collate(sample_list):

    batch_audio = [x['audio_input'] for x in sample_list]
    batch_bert_text = [x['text_input'] for x in sample_list]
    batch_asr_text = [x['asr_target'] for x in sample_list]

    #----------------tokenize and pad the audio----------------------#

    batch_audio = audio_processor(batch_audio, sampling_rate=16000).input_values

    batch_audio = [{"input_values": audio} for audio in batch_audio]
    batch_audio = audio_processor.pad(
            batch_audio,
            padding=True,
            return_tensors="pt",
        )

    with audio_processor.as_target_processor():
        label_features = audio_processor(batch_asr_text).input_ids

    label_features = [{"input_ids": labels} for labels in label_features]

    with audio_processor.as_target_processor():
        labels_batch = audio_processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

    ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    #----------------tokenize and pad the text----------------------#
    batch_text = tokenizer(batch_bert_text, padding=True, truncation=True, return_tensors="pt")
    batch_text_inputids = batch_text['input_ids']
    batch_text_attention = batch_text['attention_mask']


    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)

    #----------------tokenize and pad the extras----------------------#
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])


    return batch_text,(batch_audio,audio_length),(ctc_labels,batch_label)
