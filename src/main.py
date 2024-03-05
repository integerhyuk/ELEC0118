import argparse
import os
import re
import time

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, AutoModel

from src.config import BERT_MODEL, WAV2VEC_MODEL, BATCH_SIZE, DEVICE
from src.const import tokenizer
from src.dataset import IEMOCAPDataset, collate
from src.model import FuseModel
from src.utils import create_processor, evaluate_metrics


# bert_input = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt") # text => input_ids, attention_mask(tensor)

# dataset __get_item__()역할이 어디까지 일까?


# collate function 의 역할?

def run_infer(args, config, valid_data, checkpoint_path, session):
    ############################ PARAMETER SETTING ##########################
    num_workers = args.num_workers
    batch_size = args.batch_size

    ############################## PREPARE DATASET ##########################
    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=collate,
        shuffle=False, num_workers=num_workers
    )

    ############################## CREATE MODEL ##########################

    print("*" * 40)
    print("Create model")
    print("*" * 40)

    config_mmi = BertConfig('config.json')
    bert_model = AutoModel.from_pretrained(BERT_MODEL)
    # text_config = copy.deepcopy(roberta_model.config)

    model = FuseModel(config_mmi)

    del config_mmi

    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    bert_model.to(DEVICE)
    model.to(DEVICE)

    # ########################### INFERENCE #####################################
    print("*" * 40)
    print("Inference started")
    print("*" * 40)

    start_time = time.time()
    pred_y, true_y = [], []
    with torch.no_grad():
        time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
        for text_input, audio_input, label_input in tqdm(valid_loader):
            torch.cuda.empty_cache()
            acoustic_input, acoustic_length = audio_input[0]['input_values'].to(DEVICE), audio_input[1].to(DEVICE)
            ctc_labels, emotion_labels = label_input[0].to(DEVICE), label_input[1].to(DEVICE)
            text_input.to(DEVICE)

            true_y.extend(list(emotion_labels.cpu().numpy()))
            bert_output = bert_model(**text_input).last_hidden_state
            # logits = model(bert_output,text_input.attention_mask, acoustic_input, acoustic_length, ctc_labels, emotion_labels)
            logits = model(bert_output,text_input.attention_mask, acoustic_input, acoustic_length)

            prediction = torch.argmax(logits, axis=1)
            label_outputs = prediction.cpu().detach().numpy().astype(int)

            pred_y.extend(list(label_outputs))
            # del valid_loader

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)

        elapsed_time = time.time() - start_time
        print("The time elapse is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} '.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()])
        ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--session', default=1, type=int, help='Session index')
    parser.add_argument('--config_path',default="/home/image/MMER/configs/iemocap-ours.yaml", type=str, help='Path to config')
    parser.add_argument('--csv_path',default="/home/image/MMER/data/iemocap.csv", type=str, help='Path to csv file')
    parser.add_argument('--data_path_audio', default="/home/image/MMER/data/iemocap/", type=str, help='Path to audio data')
    # parser.add_argument('--data_path_roberta', type=str, help='Path to text data')
    parser.add_argument('--checkpoint_path', default="/home/image/MMER/output/1_model.pt", type=str, help='Path to checkpoint')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='gpu number')
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")
    args = parser.parse_args()

    config_path = args.config_path
    csv_path = args.csv_path
    data_path_audio = args.data_path_audio
    # data_path_roberta = args.data_path_roberta
    checkpoint_path = args.checkpoint_path

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    report_result = []

    df_emotion = pd.read_csv(csv_path)

    valid_session = "Ses0" + str(args.session)
    valid_data_csv = df_emotion[df_emotion["FileName"].str.match(valid_session)]
    valid_data_csv.reset_index(drop=True, inplace=True)

    valid_data = []

    for row in valid_data_csv.itertuples():
        file_name = os.path.join(args.data_path_audio + row.FileName)
        valid_data.append((file_name, row.Sentences, row.Label, row.text, row.AugmentedText))

    report_metric = run_infer(args, config, valid_data, checkpoint_path, str(args.session))
