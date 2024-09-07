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
    DataCollatorForLanguageModeling
)

import json
import pandas as pd
from peft import LoraConfig, PeftModel, get_peft_model

from dataset import TrainingDataset

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
NEW_MODEL = "Mistral-7B-v0.1-sft"
TRAIN_DATA_NAMES = ["train", "valid", "test"]

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

        logger.info("done.")
    
    logger.info("loading dataset...")
    train, val, test = [check_and_load_dataset(name) for name in TRAIN_DATA_NAMES]
    logger.info("dataset loaded.")

    return train, val, test



def train_model(args):
    """ Rock and roll """
    logger.info("starting model training...")

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
    train, val, test = load_dataset(args)
    logger.debug(train)

    train_model(args)