import os
import torch.quantization
from transformers import AutoTokenizer, AutoModelForCausalLM


def model_exists(path):
    files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json",
             "special_tokens_map.json"]
    return all(os.path.isfile(os.path.join(path, file)) for file in files)


#app_dir = 'M:/dev/python-multi/models/llm/open-assistant'
app_dir = '/models/llm/OpenAssistant/oasst-sft-6-llama-30b-xor'

# Check for an exiting version of the model and download it if is missing
if not model_exists(app_dir):
    # Download the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-6-llama-30b-xor")
    model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-6-llama-30b-xor")
    # Save the tokenizer and model to your app directory
    tokenizer.save_pretrained(app_dir)
    model.save_pretrained(app_dir)

tokenizer = AutoTokenizer.from_pretrained(app_dir)
model = AutoModelForCausalLM.from_pretrained(app_dir)

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


# Start an interactive loop for asking and answering questions
while True:
    question = input("Please enter your question (type 'exit' to quit): ")

    if question.strip().lower() == "exit":
        break

    # Tokenize the input question and generate the response
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate the output with attention_mask using quantized_model
    output_ids = quantized_model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0])

    print("Answer:", response)
