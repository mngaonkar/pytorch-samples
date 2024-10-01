import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('mps')

def main():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                                 torch_dtype="auto",
                                                 trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    input_text = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(text)

if __name__ == "__main__":
    main()



