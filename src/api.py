import librosa
import torchaudio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModel, BertConfig
import soundfile as sf
from io import BytesIO

from src.config import BERT_MODEL, WHISPER_MODEL, WAV2VEC_MODEL, idx2label
from src.model import FuseModel
from src.utils import create_processor
import cv2

app = FastAPI()

device = "mps" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
speech_model_id = WHISPER_MODEL
text_model_id = BERT_MODEL

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    speech_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
mmer_model = FuseModel(BertConfig('config.json')).to(device)
checkpoint = torch.load("output/1_model.pt", map_location=torch.device(device))
state_dict = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
mmer_model.load_state_dict(state_dict)
bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)

whisper_processor = AutoProcessor.from_pretrained(speech_model_id)
wav2vec_processor = create_processor(WAV2VEC_MODEL, type="wav2vec")
text_tokenizer = AutoProcessor.from_pretrained(text_model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav" and file.content_type != "audio/wave":
        return JSONResponse(content={"error": "This API supports only WAV audio files."}, status_code=400)

    try:
        audio_bytes = await file.read()
        audio, samplerate = librosa.load(BytesIO(audio_bytes), sr=16000)

        result = pipe({"array": audio, "sampling_rate": samplerate})

        seperated_audio_result = []
        for chunk in result['chunks']:
            if 'timestamp' not in chunk or chunk['timestamp'] is None or len(chunk['timestamp']) < 2:
                print(f"Skipping chunk due to missing or invalid timestamp: {chunk}")
                continue

            start, end = chunk['timestamp']
            if start is None or end is None:
                print(f"Skipping chunk due to missing start or end timestamp: {chunk}")
                continue

            start = int(start * samplerate)
            end = int(end * samplerate)

            try:
                audio_input = audio[start:end]
                text = chunk['text']
                emotion = get_emotion(audio_input, samplerate, text)

                seperated_audio_result.append({
                    'text': text,
                    'emotion': emotion,
                })
            except Exception as e:
                print(f"Error processing chunk: {chunk}. Error: {str(e)}")

        result['seperated_audio'] = seperated_audio_result

        return result
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return JSONResponse(content={"error": "An error occurred during transcription."}, status_code=500)

def get_emotion(audio, sampling_rate, text):
    processed_audio = wav2vec_processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_values.to(device)
    tokenized_text = text_tokenizer(text, return_tensors="pt").to(device)

    bert_output = bert_model(**tokenized_text).last_hidden_state
    logit = mmer_model(bert_output, tokenized_text.attention_mask, processed_audio, [len(audio)])
    prediction = torch.softmax(logit, dim=1).squeeze().tolist()

    return {
        "Happy": prediction[0],
        "Angry": prediction[1],
        "Neutral": prediction[2],
        "Sad": prediction[3]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)