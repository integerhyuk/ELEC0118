import librosa
import torchaudio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModel, BertConfig
import soundfile as sf
from io import BytesIO

from facerecognition.model.resnet import ResNet18
from src.config import BERT_MODEL, WHISPER_MODEL, WAV2VEC_MODEL, idx2label
from src.model import FuseModel
from src.utils import create_processor
import cv2

# Define the FastAPI app
app = FastAPI()

# MMER model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
speech_model_id = WHISPER_MODEL
text_model_id = BERT_MODEL

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    speech_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
mmer_model = FuseModel(BertConfig('config.json')).to(device)
checkpoint = torch.load("../output/1_model.pt") # checkpoint가져오기
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

# Face Recognition Model
# checkpoint = torch.load('../facerecognition/model/best_checkpoint.tar', map_location=torch.device('cpu'))
# model = ResNet18()
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Define the endpoint for audio file upload
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav" and file.content_type != "audio/wave":
        return JSONResponse(content={"error": "This API supports only WAV audio files."}, status_code=400)

    # Load the audio file using librosa
    audio_bytes = await file.read()
    audio, samplerate = librosa.load(BytesIO(audio_bytes), sr=16000)

    # Perform inference
    result = pipe({"array": audio, "sampling_rate": samplerate})

    # Optionally, extract timestamps and texts here as needed
    seperated_audio_result = []
    for chunk in result['chunks']:
        start, end = chunk['timestamp']
        start = int(start * samplerate)
        end = int(end * samplerate)

        audio_input = audio[start:end]
        text = chunk['text']
        emotion = get_emotion(audio_input, samplerate, text)

        seperated_audio_result.append({
            'text': text,
            'emotion': emotion,
        })

    result['seperated_audio'] = seperated_audio_result

    # Return the Whisper output
    return result

# @app.post("/recognize/")
# async def recognize_video(file: UploadFile = File(...), sampling_rate: int = 1):
#
#     if not file.content_type.startswith("video/"):
#         from fastapi import HTTPException
#         raise HTTPException(status_code=400, detail="This API supports only video files.")
#
#     video_bytes = await file.read()
#     frames_and_timestamps = video_to_frames_with_timestamps(video_bytes, sampling_rate)
#     grouped_averages_with_time = process_frames_and_calculate_averages(frames_and_timestamps)
#
#     return {"grouped_averages_with_time": grouped_averages_with_time}

# Return the MMER output
def get_emotion(audio, sampling_rate, text):
    processed_audio = wav2vec_processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_values.to(device)
    tokenized_text = text_tokenizer(text, return_tensors="pt").to(device)

    # Perform inference
    bert_output = bert_model(**tokenized_text).last_hidden_state
    logit = mmer_model(bert_output, tokenized_text.attention_mask, processed_audio, [len(audio)])
    prediction = torch.softmax(logit, dim=1).squeeze().tolist()

    return {
        "Happy" : prediction[0],
        "Angry" : prediction[1],
        "Neutral" : prediction[2],
        "Sad" : prediction[3]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
