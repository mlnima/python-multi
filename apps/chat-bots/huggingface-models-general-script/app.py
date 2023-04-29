import os
from transformers import AutoTokenizer, AutoModelForCausalLM
models_path = os.path.join(os.path.abspath(os.getcwd()), '../../../models')
model_name = 'OpenAssistant/stablelm-7b-sft-v7-epoch-3'
app_dir = os.path.join(models_path, model_name.replace("/", "-"))


def model_exists(path):
    files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json",
             "special_tokens_map.json"]
    return all(os.path.isfile(os.path.join(path, file)) for file in files)


def ensure_model(model_name, app_dir):
    if not model_exists(app_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(app_dir)
        model.save_pretrained(app_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(app_dir)
        model = AutoModelForCausalLM.from_pretrained(app_dir)
    return tokenizer, model


tokenizer, model = ensure_model(model_name, app_dir)

while True:
    question = input("Please enter your question (type 'exit' to quit): ")

    if question.strip().lower() == "exit":
        break

    # Tokenize the input question and generate the response
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate the output with attention_mask using the model
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1,
                                pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0])

    print("Answer:", response)