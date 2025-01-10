from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = "openai-community/gpt2-xl"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(
    model_name,
    device_map="auto",  # Automatyczna alokacja na GPU
    offload_folder="./offload_folder"  # Jeśli zabraknie pamięci GPU, model offloaduje na dysk
)

def generate_answer(question):
    prompt = (
        f"{question}"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    output = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=2,
        max_new_tokens=600,
        temperature=0.1,
        top_k=30,
        top_p=0.9,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# question = "If a person is in court because he is guilty of a crime, why is he given a lawyer to defend him in the case even though he is guilty?"
question = "why cant america just ban guns?"
answer = generate_answer(question)
print(answer)
