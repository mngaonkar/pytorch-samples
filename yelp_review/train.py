from datasets import load_dataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
import evaluate
import numpy as np

# Hyperparameters
NUM_TRAIN_SAMPLES = 10000
NUM_TEST_SAMPLES = 1000
NEW_MODEL = "yelp_review_full_model"
NUM_EPOCHS = 5
DEVICE_TYPE = "mps"
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

# Model and dataset
MODEL = "google-bert/bert-base-cased"
# DATASET = "yelp_review_full"
DATASET = "mgb-dx-meetup/product-reviews"

torch.backends.quantized.engine = 'qnnpack'

def load_dataset_from_hf(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=LEARNING_RATE,
                                           weight_decay=WEIGHT_DECAY,
                                           dropout=DROPOUT)
        return self.optimizer
    

def main():
    dataset = load_dataset_from_hf(DATASET)
    # rename stars column to label
    dataset = dataset.rename_column("stars", "label")

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                              padding=True,
                                              truncation=True,
                                              return_tensors="pt")

    def tokenize_function(examples):
        return tokenizer(examples["review_body"], 
                         truncation=True, 
                         padding=True, 
                         max_length=256, 
                         return_tensors="pt"
                         )
    
    train_dataset = dataset['train'].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(NUM_TEST_SAMPLES)) 

    # Free up memory
    dataset = None

    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)

    # train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # id2label = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    # label2id = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, 
        num_labels=5,
        # id2label=id2label,
        # label2id=label2id,
        )

    # quantized_model = torch.quantization.quantize_dynamic(
    #     model, 
    #     {torch.nn.Linear},  # Specify the layers to be quantized (e.g., Linear layers)
    #     dtype=torch.qint8    # Use 8-bit integer quantization
    # )

    quantized_model = model
    
    # Freeze all BERT layers (except the classification head)
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Freeze the first 10 layers of BERT
    # for name, param in quantized_model.named_parameters():
    #     if "encoder.layer" in name:
    #         layer_num = int(name.split(".")[3])
    #         if layer_num < 10:
    #             param.requires_grad = False

    quantized_model.to(DEVICE_TYPE)

    def compute_metrics(pred):
        metric = evaluate.load("accuracy")

        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)

        return accuracy
    
    
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      eval_strategy="epoch", 
                                      per_device_train_batch_size=TRAIN_BATCH_SIZE, 
                                      per_device_eval_batch_size=EVAL_BATCH_SIZE, 
                                      num_train_epochs=NUM_EPOCHS, 
                                      logging_dir="./logs", 
                                      logging_steps=10, 
                                      eval_steps=10, 
                                      learning_rate=LEARNING_RATE,
                                      metric_for_best_model="accuracy",
                                      report_to="tensorboard"
                                      )
    
    trainer = CustomTrainer(
        model=quantized_model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate()

    trainer.model.save_pretrained(NEW_MODEL)

if __name__ == '__main__':
    main()