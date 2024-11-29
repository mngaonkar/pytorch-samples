from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map='mps',
    trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

generate = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generate("Who is Albert Einstein,", max_length=50, num_return_sequences=5)

for item in output:
    print("------")
    print(item['generated_text'])