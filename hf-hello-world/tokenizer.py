from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
output = tokenizer("Hello, I'm a language model,", return_tensors="pt")

for id in output["input_ids"][0]:
    print(f"id = {id} token = {tokenizer.decode([id])}")

# print random token
print(f"token for id 3323 = {tokenizer.decode(3323)}")