from datasets import load_dataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import numpy as np

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 1000
NEW_MODEL = "yelp_review_full_model"
NUM_EPOCHS = 50
DEVICE_TYPE = "cuda"
TRAIN_BATCH_SIZE = 80
EVAL_BATCH_SIZE = 80
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

def load_dataset_from_hf(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        return self.optimizer
    

def main():
    dataset = load_dataset_from_hf("yelp_review_full")
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    train_dataset = dataset['train'].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(NUM_TEST_SAMPLES)) 

    # Free up memory
    dataset = None

    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)

    train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    id2label = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    label2id = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", 
        num_labels=5,
        id2label=id2label,
        label2id=label2id
        )
    # Freeze all BERT layers (except the classification head)
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # Freeze the first 10 layers of BERT
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split(".")[3])
            if layer_num < 10:
                param.requires_grad = False

    model.to(DEVICE_TYPE)

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
        model=model,
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