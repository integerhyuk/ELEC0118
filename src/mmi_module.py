# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F
from transformers import BertModel, Wav2Vec2Model, WavLMModel, HubertModel, BertLayer
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertSelfOutput

from src.utils import create_mask
from src.config import LEARNING_RATE, WAV2VEC_MODEL, BATCH_SIZE, EPOCH, ACCUM_GRAD

args = {
    "input_feat": 1024,
    "encoder_embed_dim": 1024,
    "encoder_layers": 3,
    "dropout": 0.1,
    "activation_dropout": 0,
    "dropout_input": 0.1,
    "attention_dropout": 0.1,
    "encoder_layerdrop": 0.05,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "encoder_ffn_embed_dim": 2048,
    "encoder_attention_heads": 4,
    "activation_fn": "gelu",
    "layer_norm_first": False,
}

class Config:
    pass

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertSelfEncoder(nn.Module):
    def __init__(self, config):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)

class MMI_Model(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, ctc_output_size, label_output_size, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2, auxnum_labels=2):
        #super(BertPreTrainedModel, self).__init__()
        super(MMI_Model, self).__init__()
        self.num_labels = num_labels
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL,output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        # self.wav2vec2 = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        # self.wav2vec2 = HubertModel.from_pretrained("facebook/hubert-base-ls960",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        self.wav2vec2.feature_extractor._freeze_parameters()
        #self.trans_matrix = torch.zeros(num_labels, auxnum_labels)
        # self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.img2txt_attention = BertCrossEncoder(config, layer_num2)
        self.txt2txt_attention = BertCrossEncoder(config, layer_num3)
        # self.txt2fbank_attention = BertCrossEncoder(config, 1)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.gate_fbank = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.upsample_fbank2wav = nn.Linear(160, 768)
        ### self.self_attention = BertLastSelfAttention(config)
        # self.classifier = nn.Linear(config.hidden_size * 2, label_output_size)
        self.classifier = nn.Sequential(OrderedDict([
        #   ('linear1', nn.Linear(config.hidden_size * 2, config.hidden_size)), # make *4 and *2 when using stats pooling
        #   ('relu1', nn.ReLU()),
        #   ('linear2', nn.Linear(config.hidden_size * 2, config.hidden_size)),
        #   ('relu2', nn.ReLU()),
          ('linear3', nn.Linear(config.hidden_size * 2, label_output_size))
          ]))
        # nn.Linear(config.hidden_size * 2, label_output_size)
        # self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        # self.crf = CRF(num_labels, batch_first=True)

        # config_audio = Config()

        # for arg_name, arg_val in args.items():
        #     setattr(config_audio, arg_name, arg_val)

        self.dropout_audio_input = nn.Dropout(0.1)

        # self.audio_encoder = TransformerEncoder(config_audio)
        self.ctc_linear = nn.Linear(768, ctc_output_size)

        # self.semantic_excite = nn.Linear(768, 768)
        # self.acoustic_excite = nn.Linear(768,768)
        # self.downsample_final = nn.Linear(768*4, 768*2) only in case of stats pooling
        self.downsample_final = nn.Linear(768*2, 768)

        # self.conv2d_subsample = Conv2dSubsampling2(768, 768, 0.1)
        # self.conv2d_subsample_fbank = Conv2dSubsampling(768, 768, 0.1)

        # self.aux_crf = CRF(auxnum_labels, batch_first=True)

        #self.apply(self.init_bert_weights)

        self.weights = nn.Parameter(torch.zeros(13))

        self.fuse_type = 'max'

        if self.fuse_type == 'att':
            self.output_attention_audio = nn.Sequential(
                nn.Linear(768, 768 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 // 2, 1)
            )
            self.output_attention_multimodal = nn.Sequential(
                nn.Linear(768*2, 768*2 // 2),
                ActivateFun("gelu"),
                nn.Linear(768*2 // 2, 1)
            )
            # self.output_attention_text = nn.Sequential(
            #     nn.Linear(768, 768 // 2),
            #     ActivateFun("gelu"),
            #     nn.Linear(768 // 2, 1)
            # )

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).type(torch.IntTensor)


            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                    )

        return loss

    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def _weighted_sum(self, feature, normalize):

        stacked_feature = torch.stack(feature, dim=0)

        if normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],))

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(13, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(self, bert_output, bert_attention_mask, audio_input, audio_length): #, ctc_labels, emotion_labels, augmentation = False):

        # text_output = self.bert(bert_input_ids,attention_mask=bert_attention_mask,token_type_ids=bert_segment_ids)
        # text_output = self.dropout(text_output[0])
        bert_output = self.dropout(bert_output)
        # audio_output_wav2vec2_all = self.wav2vec2(audio_input) #only in average
        # audio_output_wav2vec2 = audio_output_wav2vec2_all[0] #only in average
        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0] #imp
        # print(audio_output_wav2vec2_all)
        # print(audio_output_wav2vec2_all[2])
        # print(audio_output_wav2vec2_layers.shape)
        # print(audio_output_wav2vec2_all[2][-8:])
        # print(torch.cat(audio_output_wav2vec2_all[2][-8:], axis = -1).shape)
        # audio_output_wav2vec2_layers = torch.mean(torch.stack(audio_output_wav2vec2_all[2][-8:], axis = 0), axis = 0)
        # print(audio_output_wav2vec2_layers.shape)
        # audio_output_wav2vec2_2 = self._weighted_sum([f for f in audio_output_wav2vec2_all["hidden_states"]], True) #weighted mean
        # audio_output_wav2vec2_2 = torch.mean(torch.stack(audio_output_wav2vec2_all["hidden_states"][-8:], axis = 0), axis = 0) #8 average

        #-----------------------------------------------------------------------------------------------------------#
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        audio_attention_mask, fbank_attention_mask, wav2vec2_attention_mask, input_lengths = None, None, None, None

        audio_attention_mask = create_mask(audio_input.shape[0],audio_input.shape[1],audio_length)

        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).type(torch.IntTensor)
        wav2vec2_attention_mask = create_mask(audio_output_wav2vec2.shape[0],audio_output_wav2vec2.shape[1],input_lengths)

        wav2vec2_attention_mask = wav2vec2_attention_mask.to(bert_output.device)

        #-----------------------------------------------------------------------------------------------------------#

        audio_output_dropout = self.dropout_audio_input(audio_output_wav2vec2)
        # logits_ctc = self.ctc_linear(audio_output_dropout)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        main_addon_sequence_encoder = self.self_attention_v2(bert_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1][-1]

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # subsample the frames to 1/4th of the number
        audio_output = audio_output_wav2vec2.clone()
        # audio_output, wav2vec2_attention_mask = self.conv2d_subsample(audio_output_wav2vec2,wav2vec2_attention_mask.unsqueeze(1)) # remove _2

        # Exclusively for fbank
        # fbank_input = self.upsample_fbank2wav(fbank_input)
        # audio_output_fbank, fbank_attention_mask = self.conv2d_subsample_fbank(fbank_input,fbank_attention_mask.unsqueeze(1))

        # project audio embeddings to a smaller space
        converted_vis_embed_map = self.vismap2text(audio_output)

        #--------------------applying txt2img attention mechanism to obtain image-based text representations----------------------------#

        # calculate added attention mask
        # img_mask = added_attention_mask = torch.ones([audio_output.shape[0],audio_output.shape[1]]).cuda()

        # calculate added attention mask
        img_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        # calculate extended_fbank_mask required for cross-attention
        # fbank_attention_mask = fbank_attention_mask.squeeze(1)
        # extended_fbank_mask = fbank_attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_fbank_mask = extended_fbank_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_fbank_mask = (1.0 - extended_fbank_mask) * -10000.0

        # print(main_addon_sequence_output.shape)
        # print(converted_vis_embed_map.shape)
        # print(extended_img_mask.shape)

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim


        # exclusively for fbank
        # cross_encoder_fbank = self.txt2fbank_attention(main_addon_sequence_output, audio_output_fbank, extended_fbank_mask)
        # cross_output_layer_fbank = cross_encoder_fbank[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#


        #----------------------apply img2txt attention mechanism to obtain multimodal-based text representations-------------------------#

        # project audio embeddings to a smaller space || left part of the image
        converted_vis_embed_map_v2 = self.vismap2text_v2(audio_output)

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * audio_length * hidden_dim

                                            #----------------------------------#

        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#

        #---------------------------------------apply visual gate and get final representations---------------------------------------------#

        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        #exclusively for fbank
        # merge_representation_fbank = torch.cat((cross_final_txt_layer, cross_output_layer_fbank), dim=-1)
        # gate_value_fbank = torch.sigmoid(self.gate_fbank(merge_representation_fbank))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed_fbank = torch.mul(gate_value_fbank, cross_output_layer_fbank)

        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed, gated_converted_att_vis_embed_fbank), dim=-1)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        # Keep acoustic as importance
        # merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed = torch.mul(gate_value, cross_final_txt_layer)
        # final_output = torch.cat((cross_output_layer, gated_converted_att_vis_embed), dim=-1)

        # # if self excitation
        # semantic_excit = F.sigmoid(self.semantic_excite(cross_output_layer))
        # semantic_embed = cross_final_txt_layer * semantic_excit + cross_final_txt_layer # These two lines are different, we add the residual connection
        # acoustic_excit = F.sigmoid(self.acoustic_excite(cross_final_txt_layer))
        # acoustic_embed = cross_output_layer * acoustic_excit + cross_output_layer # These two lines are different, we add the residual connection
        # final_output = torch.cat([semantic_embed,acoustic_embed],dim=2)


        # classification_feats_pooled = torch.mean(final_output, dim=1)
        # classification_feats_pooled = self.classifier(classification_feats_pooled)

        # classification_feats_audio = torch.mean(audio_output, dim=1)
        # classification_feats_multimodal = torch.mean(final_output, dim=1)

        # # stats pooling
        # classification_feats_audio = torch.cat((torch.mean(audio_output,dim=1),torch.std(audio_output,dim=1)), dim=-1) #768*2
        # classification_feats_multimodal = torch.cat((torch.mean(final_output,dim=1),torch.std(final_output,dim=1)), dim=-1) #768*4
        # classification_feats_multimodal = self.downsample_final(classification_feats_multimodal) #768*2
        # final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1) #768*4
        # classification_feats_pooled = self.classifier(final_output)

        # audio_attention_mask = None
        # audio_attention_mask = create_mask(audio_input.shape[0],audio_input.shape[1],audio_length)

        audio_output_pool = audio_output_wav2vec2.clone() #change _2
        # audio_output_pool = audio_output.clone()

        multimodal_output = final_output.clone()

        text_output_2 = bert_output.clone()

        if self.fuse_type == 'mean':
            if audio_attention_mask is None:
                classification_feats_audio = torch.mean(audio_output_wav2vec2, dim=1)
            else:
                padding_mask = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1], audio_attention_mask)
                padding_mask = padding_mask.to(audio_output_wav2vec2.device)
                audio_output_pool[~padding_mask] = 0.0 #mean
                classification_feats_audio = audio_output_pool.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1) #mean
        elif self.fuse_type == 'max':
            padding_mask = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1], audio_attention_mask)
            padding_mask = padding_mask.to(audio_output_wav2vec2.device)
            audio_output_pool[~padding_mask] = -9999.9999 #max
            classification_feats_audio,_ = torch.max(audio_output_pool,dim = 1) #max
        elif self.fuse_type == 'att':
            text_image_mask = wav2vec2_attention_mask_back.permute(1, 0).contiguous()
            # text_image_mask = wav2vec2_attention_mask.squeeze(1).permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:audio_output_pool.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention_audio(audio_output_pool)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
            classification_feats_audio = (text_image_alpha.unsqueeze(-1) * audio_output_pool).sum(dim=1)
        elif self.fuse_type == 'stats':
            classification_feats_audio = torch.cat((torch.mean(audio_output_pool,dim=1),torch.std(audio_output_pool,dim=1)), dim=-1) #768*2


        if self.fuse_type == 'mean':
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = 0.0 #mean
            classification_feats_multimodal = multimodal_output.sum(dim=1) / padding_mask_text.sum(dim=1).view(-1, 1) #mean
            # classification_feats_multimodal = torch.mean(final_output, dim=1)
        elif self.fuse_type == 'max':
            padding_mask_text = bert_attention_mask > 0
            multimodal_output[~padding_mask_text] = -9999.9999 #max
            classification_feats_multimodal,_ = torch.max(multimodal_output,dim = 1) #max
        elif self.fuse_type == 'att':
            multimodal_mask = bert_attention_mask.permute(1, 0).contiguous()
            multimodal_mask = multimodal_mask[0:multimodal_output.size(1)]
            multimodal_mask = multimodal_mask.permute(1, 0).contiguous()

            multimodal_alpha = self.output_attention_multimodal(multimodal_output)
            multimodal_alpha = multimodal_alpha.squeeze(-1).masked_fill(multimodal_mask == 0, -1e9)
            multimodal_alpha = torch.softmax(multimodal_alpha, dim=-1)
            classification_feats_multimodal = (multimodal_alpha.unsqueeze(-1) * multimodal_output).sum(dim=1)
        elif self.fuse_type == 'stats':
            classification_feats_multimodal = torch.cat((torch.mean(multimodal_output,dim=1),torch.std(multimodal_output,dim=1)), dim=-1)

        # if self.fuse_type == 'mean':
        #     padding_mask_text = bert_attention_mask > 0
        #     text_output_2[~padding_mask_text] = 0.0
        #     classification_feats_text = text_output_2.sum(dim=1) / padding_mask_text.sum(dim=1).view(-1, 1)
        #     # classification_feats_multimodal = torch.mean(final_output, dim=1)
        # if self.fuse_type == 'att':

        #     text_mask = bert_attention_mask.permute(1, 0).contiguous()
        #     text_mask = text_mask[0:multimodal_output.size(1)]
        #     text_mask = text_mask.permute(1, 0).contiguous()

        #     text_alpha = self.output_attention_text(text_output_2)
        #     text_alpha = text_alpha.squeeze(-1).masked_fill(text_mask == 0, -1e9)
        #     text_alpha = torch.softmax(text_alpha, dim=-1)
        #     classification_feats_text = (text_alpha.unsqueeze(-1) * text_output_2).sum(dim=1)


        classification_feats_multimodal = self.downsample_final(classification_feats_multimodal)
        final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1)
        classification_feats_pooled = self.classifier(final_output)

        #------------------------------------------------------------------------------------------------------------------------------------#

        #------------------------------------------------------calculate losses---------------------------------------------------------------#
        #
        # loss = None
        # loss_ctc = None
        # loss_cls = None
        # if not augmentation:
        #     loss_ctc = self._ctc_loss(logits_ctc, ctc_labels, input_lengths, audio_attention_mask) #ctc loss
        #     loss_cls = self._cls_loss(classification_feats_pooled, emotion_labels) #cls loss

        return classification_feats_pooled, final_output #, loss_cls, loss_ctc

class MMI_Model_SingleLoss(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, ctc_output_size, label_output_size, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2, auxnum_labels=2):
        #super(BertPreTrainedModel, self).__init__()
        super(MMI_Model_SingleLoss, self).__init__()
        self.num_labels = num_labels
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL,output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        self.wav2vec2.feature_extractor._freeze_parameters()
        #self.trans_matrix = torch.zeros(num_labels, auxnum_labels)
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(768, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(768, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.img2txt_attention = BertCrossEncoder(config, layer_num2)
        self.txt2txt_attention = BertCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        ### self.self_attention = BertLastSelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size * 2, label_output_size)
        # self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        # self.crf = CRF(num_labels, batch_first=True)

        # config_audio = Config()

        # for arg_name, arg_val in args.items():
        #     setattr(config_audio, arg_name, arg_val)

        self.dropout_audio_input = nn.Dropout(0.1)

        # self.audio_encoder = TransformerEncoder(config_audio)
        # self.ctc_linear = nn.Linear(768, ctc_output_size)

        # self.semantic_excite = nn.Linear(768, 768)
        # self.acoustic_excite = nn.Linear(768, 768)
        self.downsample_final = nn.Linear(768*2, 768)

        # self.aux_crf = CRF(auxnum_labels, batch_first=True)

        #self.apply(self.init_bert_weights)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            # input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                    )

        return loss

    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def forward(self, bert_input_ids, bert_attention_mask, bert_segment_ids, text_length, audio_input, audio_length, ctc_labels, emotion_labels, text_output):


        # text_output = self.bert(bert_input_ids,attention_mask=bert_attention_mask,token_type_ids=bert_segment_ids)
        # text_output = self.dropout(text_output[0])
        text_output = self.dropout(text_output)

        audio_output = self.wav2vec2(audio_input)
        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(torch.ones_like(audio_input).sum(-1)).type(torch.IntTensor)
        audio_output = self.dropout_audio_input(audio_output[0])
        # logits_ctc = self.ctc_linear(audio_output)

        extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)

        main_addon_sequence_encoder = self.self_attention_v2(text_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]


        # project audio embeddings to a smaller space
        converted_vis_embed_map = self.vismap2text(audio_output)

        #--------------------applying txt2img attention mechanism to obtain image-based text representations----------------------------#

        # calculate added attention mask
        img_mask = added_attention_mask = torch.ones([audio_output.shape[0],audio_output.shape[1]]).cuda()

        # calculate extended_img_mask required for cross-attention
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        # print(main_addon_sequence_output.shape)
        # print(converted_vis_embed_map.shape)
        # print(extended_img_mask.shape)

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#


        #----------------------apply img2txt attention mechanism to obtain multimodal-based text representations-------------------------#

        # project audio embeddings to a smaller space || left part of the image
        converted_vis_embed_map_v2 = self.vismap2text_v2(audio_output)

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * audio_length * hidden_dim

                                            #----------------------------------#

        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#

        #---------------------------------------apply visual gate and get final representations---------------------------------------------#

        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        # Keep acoustic as importance
        # merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed = torch.mul(gate_value, cross_final_txt_layer)
        # final_output = torch.cat((cross_output_layer, gated_converted_att_vis_embed), dim=-1)

        # if self excitation
        # semantic_excit = F.sigmoid(self.semantic_excite(cross_output_layer))
        # semantic_embed = cross_final_txt_layer * semantic_excit + cross_final_txt_layer # These two lines are different, we add the residual connection
        # acoustic_excit = F.sigmoid(self.acoustic_excite(cross_final_txt_layer))
        # acoustic_embed = cross_output_layer * acoustic_excit + cross_output_layer # These two lines are different, we add the residual connection
        # final_output = torch.cat([semantic_embed,acoustic_embed],dim=2)


        # classification_feats_pooled = torch.mean(final_output, dim=1)
        # classification_feats_pooled = self.classifier(classification_feats_pooled)

        classification_feats_audio = torch.mean(audio_output, dim=1)
        classification_feats_multimodal = torch.mean(final_output, dim=1)
        classification_feats_multimodal = self.downsample_final(classification_feats_multimodal)
        final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1)
        classification_feats_pooled = self.classifier(final_output)

        #------------------------------------------------------------------------------------------------------------------------------------#

        #------------------------------------------------------calculate losses---------------------------------------------------------------#

        # loss_ctc = self._ctc_loss(logits_ctc, ctc_labels, input_lengths) #ctc loss

        loss_cls = self._cls_loss(classification_feats_pooled, emotion_labels) #cls loss

        loss = loss_cls

        return loss, classification_feats_pooled, loss_cls, 0.0

class MMI_Model_LateFusion(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, ctc_output_size, label_output_size, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2, auxnum_labels=2):
        #super(BertPreTrainedModel, self).__init__()
        super(MMI_Model_LateFusion, self).__init__()
        self.num_labels = num_labels
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL,output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        # self.wav2vec2 = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv",output_hidden_states=True,return_dict=True,apply_spec_augment=False)
        self.wav2vec2.feature_extractor._freeze_parameters()
        #self.trans_matrix = torch.zeros(num_labels, auxnum_labels)
        # self.self_attention = BertSelfEncoder(config)
        # self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.vismap2text = nn.Linear(768, config.hidden_size)
        # self.vismap2text_v2 = nn.Linear(768, config.hidden_size)
        # self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        # self.img2txt_attention = BertCrossEncoder(config, layer_num2)
        # self.txt2txt_attention = BertCrossEncoder(config, layer_num3)
        # self.txt2fbank_attention = BertCrossEncoder(config, 1)
        # self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.gate_fbank = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.upsample_fbank2wav = nn.Linear(160, 768)
        ### self.self_attention = BertLastSelfAttention(config)
        # self.classifier = nn.Linear(config.hidden_size * 2, label_output_size)
        self.classifier = nn.Sequential(OrderedDict([
        #   ('linear1', nn.Linear(config.hidden_size * 2, config.hidden_size)), # make *4 and *2 when using stats pooling
        #   ('relu1', nn.ReLU()),
        #   ('linear2', nn.Linear(config.hidden_size * 2, config.hidden_size)),
        #   ('relu2', nn.ReLU()),
          ('linear3', nn.Linear(config.hidden_size * 2, label_output_size))
          ]))
        # nn.Linear(config.hidden_size * 2, label_output_size)
        # self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        # self.crf = CRF(num_labels, batch_first=True)

        # config_audio = Config()

        # for arg_name, arg_val in args.items():
        #     setattr(config_audio, arg_name, arg_val)

        self.dropout_audio_input = nn.Dropout(0.1)

        # self.audio_encoder = TransformerEncoder(config_audio)
        # self.ctc_linear = nn.Linear(768, ctc_output_size)

        # self.semantic_excite = nn.Linear(768, 768)
        # self.acoustic_excite = nn.Linear(768,768)
        # self.downsample_final = nn.Linear(768*4, 768*2) only in case of stats pooling
        self.downsample_final = nn.Linear(768*2, 768)

        # self.conv2d_subsample = Conv2dSubsampling2(768, 768, 0.1)
        # self.conv2d_subsample_fbank = Conv2dSubsampling(768, 768, 0.1)

        # self.aux_crf = CRF(auxnum_labels, batch_first=True)

        #self.apply(self.init_bert_weights)

        self.fuse_type = 'att'

        if self.fuse_type == 'att':
            self.output_attention_audio = nn.Sequential(
                nn.Linear(768, 768 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 // 2, 1)
            )
            # self.output_attention_multimodal = nn.Sequential(
            #     nn.Linear(768*2, 768*2 // 2),
            #     ActivateFun("gelu"),
            #     nn.Linear(768*2 // 2, 1)
            # )
            self.output_attention_text = nn.Sequential(
                nn.Linear(768, 768 // 2),
                ActivateFun("gelu"),
                nn.Linear(768 // 2, 1)
            )

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.

    def _ctc_loss(self, logits, labels, input_lengths, attention_mask=None):

        loss = None
        if labels is not None:

            # # retrieve loss input_lengths from attention_mask
            # attention_mask = (
            #     attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            # )
            if attention_mask is not None:
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).type(torch.IntTensor)


            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                    )

        return loss

    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def forward(self, bert_input_ids, bert_attention_mask, bert_segment_ids, text_length, audio_input, audio_length, ctc_labels, emotion_labels, text_output, fbank_input, fbank_length):


        # text_output = self.bert(bert_input_ids,attention_mask=bert_attention_mask,token_type_ids=bert_segment_ids)
        # text_output = self.dropout(text_output[0])
        text_output = self.dropout(text_output)
        # audio_output_wav2vec2_all = self.wav2vec2(audio_input) #only in average
        # audio_output_wav2vec2 = audio_output_wav2vec2_all[0] #only in average
        audio_output_wav2vec2 = self.wav2vec2(audio_input)[0]
        # print(audio_output_wav2vec2_all)
        # print(audio_output_wav2vec2_all[2])
        # print(audio_output_wav2vec2_layers.shape)
        # print(audio_output_wav2vec2_all[2][-8:])
        # print(torch.cat(audio_output_wav2vec2_all[2][-8:], axis = -1).shape)
        # audio_output_wav2vec2_layers = torch.mean(torch.stack(audio_output_wav2vec2_all[2][-8:], axis = 0), axis = 0)
        # print(audio_output_wav2vec2_layers.shape)

        #-----------------------------------------------------------------------------------------------------------#
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        # create raw audio, FBank and wav2vec2 hidden state attention masks
        audio_attention_mask, fbank_attention_mask, wav2vec2_attention_mask, input_lengths = None, None, None, None

        audio_attention_mask = create_mask(audio_input.shape[0],audio_input.shape[1],audio_length)
        fbank_attention_mask = create_mask(fbank_input.shape[0],fbank_input.shape[1],fbank_length)

        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).type(torch.IntTensor)
        wav2vec2_attention_mask = create_mask(audio_output_wav2vec2.shape[0],audio_output_wav2vec2.shape[1],input_lengths)

        fbank_attention_mask = fbank_attention_mask.cuda()
        wav2vec2_attention_mask = wav2vec2_attention_mask.cuda()

        assert torch.all(wav2vec2_attention_mask.eq(fbank_attention_mask)),"wav2vec2 and FBank attention masks don't match"

        #-----------------------------------------------------------------------------------------------------------#

        # audio_output_dropout = self.dropout_audio_input(audio_output_wav2vec2)
        # logits_ctc = self.ctc_linear(audio_output_dropout)

        # extended_txt_mask = bert_attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0

        # main_addon_sequence_encoder = self.self_attention_v2(text_output, extended_txt_mask)
        # main_addon_sequence_output = main_addon_sequence_encoder[-1]

        wav2vec2_attention_mask_back = wav2vec2_attention_mask.clone()
        # # subsample the frames to 1/4th of the number
        # audio_output, wav2vec2_attention_mask = self.conv2d_subsample(audio_output_wav2vec2,wav2vec2_attention_mask.unsqueeze(1))

        # # Exclusively for fbank
        # # fbank_input = self.upsample_fbank2wav(fbank_input)
        # # audio_output_fbank, fbank_attention_mask = self.conv2d_subsample_fbank(fbank_input,fbank_attention_mask.unsqueeze(1))

        # # project audio embeddings to a smaller space
        # converted_vis_embed_map = self.vismap2text(audio_output)

        #--------------------applying txt2img attention mechanism to obtain image-based text representations----------------------------#

        # calculate added attention mask
        # img_mask = added_attention_mask = torch.ones([audio_output.shape[0],audio_output.shape[1]]).cuda()

        # calculate added attention mask
        # img_mask = wav2vec2_attention_mask.squeeze(1).clone()
        # # calculate extended_img_mask required for cross-attention
        # extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        # extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        # calculate extended_fbank_mask required for cross-attention
        # fbank_attention_mask = fbank_attention_mask.squeeze(1)
        # extended_fbank_mask = fbank_attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_fbank_mask = extended_fbank_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_fbank_mask = (1.0 - extended_fbank_mask) * -10000.0

        # print(main_addon_sequence_output.shape)
        # print(converted_vis_embed_map.shape)
        # print(extended_img_mask.shape)

        # cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        # cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim


        # exclusively for fbank
        # cross_encoder_fbank = self.txt2fbank_attention(main_addon_sequence_output, audio_output_fbank, extended_fbank_mask)
        # cross_output_layer_fbank = cross_encoder_fbank[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#


        #----------------------apply img2txt attention mechanism to obtain multimodal-based text representations-------------------------#

        # project audio embeddings to a smaller space || left part of the image
        # converted_vis_embed_map_v2 = self.vismap2text_v2(audio_output)

        # cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        # cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * audio_length * hidden_dim

        #                                     #----------------------------------#

        # cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        # cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim

        #----------------------------------------------------------------------------------------------------------------------------------#

        #---------------------------------------apply visual gate and get final representations---------------------------------------------#

        # merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        #exclusively for fbank
        # merge_representation_fbank = torch.cat((cross_final_txt_layer, cross_output_layer_fbank), dim=-1)
        # gate_value_fbank = torch.sigmoid(self.gate_fbank(merge_representation_fbank))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed_fbank = torch.mul(gate_value_fbank, cross_output_layer_fbank)

        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed, gated_converted_att_vis_embed_fbank), dim=-1)
        # final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)

        # Keep acoustic as importance
        # merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        # gated_converted_att_vis_embed = torch.mul(gate_value, cross_final_txt_layer)
        # final_output = torch.cat((cross_output_layer, gated_converted_att_vis_embed), dim=-1)

        # # if self excitation
        # semantic_excit = F.sigmoid(self.semantic_excite(cross_output_layer))
        # semantic_embed = cross_final_txt_layer * semantic_excit + cross_final_txt_layer # These two lines are different, we add the residual connection
        # acoustic_excit = F.sigmoid(self.acoustic_excite(cross_final_txt_layer))
        # acoustic_embed = cross_output_layer * acoustic_excit + cross_output_layer # These two lines are different, we add the residual connection
        # final_output = torch.cat([semantic_embed,acoustic_embed],dim=2)


        # classification_feats_pooled = torch.mean(final_output, dim=1)
        # classification_feats_pooled = self.classifier(classification_feats_pooled)

        # classification_feats_audio = torch.mean(audio_output, dim=1)
        # classification_feats_multimodal = torch.mean(final_output, dim=1)

        # # stats pooling
        # classification_feats_audio = torch.cat((torch.mean(audio_output,dim=1),torch.std(audio_output,dim=1)), dim=-1) #768*2
        # classification_feats_multimodal = torch.cat((torch.mean(final_output,dim=1),torch.std(final_output,dim=1)), dim=-1) #768*4
        # classification_feats_multimodal = self.downsample_final(classification_feats_multimodal) #768*2
        # final_output = torch.cat((classification_feats_audio, classification_feats_multimodal), dim=-1) #768*4
        # classification_feats_pooled = self.classifier(final_output)

        # audio_attention_mask = None
        # audio_attention_mask = create_mask(audio_input.shape[0],audio_input.shape[1],audio_length)

        audio_output_pool = audio_output_wav2vec2.clone()

        # multimodal_output = final_output.clone()

        text_output_2 = text_output.clone()

        if self.fuse_type == 'mean':
            if audio_attention_mask is None:
                classification_feats_audio = torch.mean(audio_output_wav2vec2, dim=1)
            else:
                padding_mask = self.wav2vec2._get_feature_vector_attention_mask(audio_output_wav2vec2.shape[1], audio_attention_mask)
                padding_mask = padding_mask.to(audio_output_wav2vec2.device)
                audio_output_pool[~padding_mask] = 0.0
                classification_feats_audio = audio_output_pool.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        elif self.fuse_type == 'att':

            text_image_mask = wav2vec2_attention_mask_back.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:audio_output_pool.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention_audio(audio_output_pool)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
            classification_feats_audio = (text_image_alpha.unsqueeze(-1) * audio_output_pool).sum(dim=1)

        # if self.fuse_type == 'mean':
        #     padding_mask_text = bert_attention_mask > 0
        #     multimodal_output[~padding_mask_text] = 0.0
        #     classification_feats_multimodal = multimodal_output.sum(dim=1) / padding_mask_text.sum(dim=1).view(-1, 1)
        #     # classification_feats_multimodal = torch.mean(final_output, dim=1)
        # elif self.fuse_type == 'att':
        #     multimodal_mask = bert_attention_mask.permute(1, 0).contiguous()
        #     multimodal_mask = multimodal_mask[0:multimodal_output.size(1)]
        #     multimodal_mask = multimodal_mask.permute(1, 0).contiguous()

        #     multimodal_alpha = self.output_attention_multimodal(multimodal_output)
        #     multimodal_alpha = multimodal_alpha.squeeze(-1).masked_fill(multimodal_mask == 0, -1e9)
        #     multimodal_alpha = torch.softmax(multimodal_alpha, dim=-1)
        #     classification_feats_multimodal = (multimodal_alpha.unsqueeze(-1) * multimodal_output).sum(dim=1)


        if self.fuse_type == 'mean':
            padding_mask_text = bert_attention_mask > 0
            text_output_2[~padding_mask_text] = 0.0
            classification_feats_text = text_output_2.sum(dim=1) / padding_mask_text.sum(dim=1).view(-1, 1)
            # classification_feats_multimodal = torch.mean(final_output, dim=1)
        if self.fuse_type == 'att':

            text_mask = bert_attention_mask.permute(1, 0).contiguous()
            text_mask = text_mask[0:text_output_2.size(1)]
            text_mask = text_mask.permute(1, 0).contiguous()

            text_alpha = self.output_attention_text(text_output_2)
            text_alpha = text_alpha.squeeze(-1).masked_fill(text_mask == 0, -1e9)
            text_alpha = torch.softmax(text_alpha, dim=-1)
            classification_feats_text = (text_alpha.unsqueeze(-1) * text_output_2).sum(dim=1)


        # classification_feats_multimodal = self.downsample_final(classification_feats_multimodal)
        final_output = torch.cat((classification_feats_audio,classification_feats_text), dim=-1)
        classification_feats_pooled = self.classifier(final_output)

        #------------------------------------------------------------------------------------------------------------------------------------#

        #------------------------------------------------------calculate losses---------------------------------------------------------------#

        # loss_ctc = self._ctc_loss(logits_ctc, ctc_labels, input_lengths, audio_attention_mask) #ctc loss

        loss_cls = self._cls_loss(classification_feats_pooled, emotion_labels) #cls loss

        loss = loss_cls

        return loss, classification_feats_pooled, 0.0, loss_cls, 0.0
