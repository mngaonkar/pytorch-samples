from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "yelp_review_full_model"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("mps")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Inferencing
text = "Movie is average."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs.to("mps")
output = model(**inputs)

prediction = torch.argmax(output.logits).item()
print(f"Rating: {prediction}")