import re

from transformers import AutoTokenizer

from src.config import BERT_MODEL
from src.config import WAV2VEC_MODEL

from src.utils import create_processor

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
audio_processor = create_processor(WAV2VEC_MODEL, type="wav2vec")

vocabulary_chars_str = "".join(t for t in audio_processor.tokenizer.get_vocab().keys() if len(t) == 1)
vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if audio_processor.tokenizer.do_lower_case else 0,
    )


