from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import numpy as np

NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 100

def load_dataset_from_hf(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


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
    model.to("mps")

    def compute_metrics(pred):
        metric = evaluate.load("accuracy")

        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        print(accuracy)
        return accuracy
    
    
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      eval_strategy="epoch", 
                                      per_device_train_batch_size=8, 
                                      per_device_eval_batch_size=8, 
                                      num_train_epochs=5, 
                                      logging_dir="test_trainer/logs", 
                                      logging_steps=10, 
                                      eval_steps=10, 
                                    #   learning_rate=1e-5,
                                      metric_for_best_model="accuracy",
                                      )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == '__main__':
    main()