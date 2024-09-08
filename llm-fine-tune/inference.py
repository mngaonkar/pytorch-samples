import logging

from transformers import (
    AutoTokenizer,
    pipeline,
)

from constants import (
    BASE_MODEL,
    NEW_MODEL,
    TRAIN_DATA_NAMES,
    STOP_TOKEN,
    HUMAN_TOKEN,
    BOT_TOKEN,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("SFT model output: ")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

prompt = "table: 1-10015132-16 columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team Q: What is terrence ross' nationality A: "
pipeline = pipeline(task="text-generation", model=NEW_MODEL, tokenizer=tokenizer, max_length=200)
result = pipeline(prompt)
print(result[0]['generated_text'])