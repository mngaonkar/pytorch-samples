import numpy as np
import torch
from datasets import Dataset
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    pipeline,
    logging,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

import json
import pandas as pd
from peft import LoraConfig, PeftModel, get_peft_model

from dataset import TrainingDataset

from constants import (
    BASE_MODEL,
    NEW_MODEL,
    TRAIN_DATA_NAMES,
    STOP_TOKEN,
    HUMAN_TOKEN,
    BOT_TOKEN,
)

def load_dataset(args):
    """Load dataset from local folder """

    data_set_path = args.data
    
    def check_and_load_dataset(name):
        data_path = Path(args.data) / f"{name}.jsonl"
        logger.info(f"loading {data_path}..")

        try:
            dataset = TrainingDataset(data_path)
        except Exception as e:
            logger.critical(f"Failed to load data {data_path}")
            raise

        logger.info(f"loaded {len(dataset)} entries.")
        logger.info("done.")

        return dataset[:]
    
    logger.info("loading dataset...")
    train, val, test = [check_and_load_dataset(name) for name in TRAIN_DATA_NAMES]
    logger.info("dataset loaded.")

    return train, val, test


def train_model(args, training_dataset):
    """ Rock and roll """
    logger.info("starting model training...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Initialized model
    logger.info("configuring model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map={"":0},
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    logger.info("model configured.")

    logger.info("configuring tokenizer...")
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.resize_token_embeddings(len(tokenizer))
    logger.info("tokenizer configuration done")

    logger.info("pre SFT model output: ")
    prompt = "table: 1-10015132-16 columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team Q: What is terrence ross' nationality A: "
    old_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = old_pipeline(prompt)
    print(result[0]['generated_text'])

    # Configure LoRA
    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_args)

    model.print_trainable_parameters()

    # Configure training parameters
    training_params = TrainingArguments(
        # output_dir="./output",
        # logging_dir="./logs",
        # num_train_epochs = args.iters,
        # per_device_train_batch_size=2,
        # per_device_eval_batch_size=4,
        # gradient_accumulation_steps=16,
        # logging_steps=5,
        # learning_rate=2e-4,
        # load_best_model_at_end=True,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # warmup_ratio=0.1,
        # lr_scheduler_type="cosine",

        output_dir='./output',
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.iters,
        weight_decay=0.01
    )

    train = training_dataset["train"]
    valid = training_dataset["valid"]

    train = list(map(lambda x: tokenizer(x, truncation=True, padding='max_length'), train))
    valid = list(map(lambda x: tokenizer(x, truncation=True, padding='max_length'), valid))

    logger.info(f"train dataset length = {len(train)}")

    logger.info("sample tokenized value:")
    print(train[10])

    data_collator = data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,                  # Set to True for Masked Language Modeling
        mlm_probability=0.15,      # Probability of masking tokens
        pad_to_multiple_of=8       # Pad to a multiple of 8 for better performance
    )

    # Configure trainer
    trainer = Trainer(
        model=model,
        args=training_params,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Rock n Roll
    logger.info("starting training...")
    trainer.train()
    trainer.model.save_pretrained(NEW_MODEL)

    logger.info("model training completed.")
    

def build_parser():
    """ Build parser for command line arguments """

    parser = argparse.ArgumentParser(description="LoRA/QLoRa fine tuning")
    parser.add_argument(
        "--model",
        default="hf_model",
        help="Path for local model or huggingface repo name"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with train, valid, test subfolder and respective JSONL files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="No. of training iterations"
    )

    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    logger.debug(args)
    train, valid, test = load_dataset(args)
    training_dataset = {"train": train, "valid": valid, "test":test}
    train_model(args, training_dataset)