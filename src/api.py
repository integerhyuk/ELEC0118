from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
from io import BytesIO

# Define the FastAPI app
app = FastAPI()

# Initialize Whisper model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-small"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Define the endpoint for audio file upload
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        return JSONResponse(content={"error": "This API supports only WAV audio files."}, status_code=400)

    # Load the audio file
    audio_bytes = await file.read()
    audio, samplerate = sf.read(BytesIO(audio_bytes))

    # Perform inference
    result = pipe({"array": audio, "sampling_rate": samplerate})

    # Optionally, extract timestamps and texts here as needed

    # Return the Whisper output
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
