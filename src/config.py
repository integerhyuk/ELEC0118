# Hyperparameter tuning
LEARNING_RATE: float = 5e-4
BATCH_SIZE: int = 2
ACCUM_GRAD: int = 4
EPOCH: int = 100
#WAV2VEC_MODEL = "facebook/wav2vec2-large-xlsr-53"
WAV2VEC_MODEL: str = "facebook/wav2vec2-base-960h"
#WAV2VEC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
# BERT_MODEL = "FacebookAI/roberta-large"
BERT_MODEL: str = "microsoft/deberta-v3-base"
# BERT_MODEL: str = "microsoft/deberta-v3-large"
WHISPER = "whisper"
WAV2VEC = "wav2vec"
DEVICE = "cuda:0"
