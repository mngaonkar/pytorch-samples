from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

def load_dataset_from_hf(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


def main():
    dataset = load_dataset_from_hf("yelp_review_full")
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    model.to("mps")

    def compute_metrics(pred):
        metric = evaluate.load("accuracy")

        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      eval_strategy="epoch", 
                                      per_device_train_batch_size=8, 
                                      per_device_eval_batch_size=8, 
                                      num_train_epochs=3, 
                                      logging_dir="test_trainer/logs", 
                                      logging_steps=10, 
                                      eval_steps=10, 
                                    #   load_best_model_at_end=True, 
                                      metric_for_best_model="accuracy", 
                                      greater_is_better=True)
    
    train = tokenized_datasets["train"]
    test = tokenized_datasets["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    main()