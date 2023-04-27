import os
from transformers import AutoTokenizer, AutoModel


def model_exists(path):
    files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json",
             "special_tokens_map.json"]
    return all(os.path.isfile(os.path.join(path, file)) for file in files)


app_dir = 'M:/dev/python-multi/models/llm/chatglm'
# app_dir = os.path.join(os.getcwd(), "models", "llm", "chatglm")
if not model_exists(app_dir):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    tokenizer.save_pretrained(app_dir)
    model.save_pretrained(app_dir)

# Add the trust_remote_code=True option here
tokenizer = AutoTokenizer.from_pretrained(app_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(app_dir, trust_remote_code=True).half().cuda()

while True:
    question = input("Please enter your question (type 'exit' to quit): ")

    if question.strip().lower() == "exit":
        break

    # Tokenize the input question and generate the response
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    # Generate the output with attention_mask
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1,
                                pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0])

    print("Answer:", response)
