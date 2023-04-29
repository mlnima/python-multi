import os

def model_exists(path):
    files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json",
             "special_tokens_map.json"]
    return all(os.path.isfile(os.path.join(path, file)) for file in files)
