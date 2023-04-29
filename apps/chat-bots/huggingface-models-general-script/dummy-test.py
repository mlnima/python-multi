import os
from models_list import model_names
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, \
    AutoModelForQuestionAnswering


def choose_model(model_names):
    print("Available models:")
    for i, model_name in enumerate(model_names, start=1):
        print(f"{i}. {model_name}")

    choice = -1
    while choice not in range(1, len(model_names) + 1):
        choice = int(input("Please choose a model by entering its number: "))

    return model_names[choice - 1]


def model_exists(path):
    files = ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt", "tokenizer_config.json",
             "special_tokens_map.json"]
    return all(os.path.isfile(os.path.join(path, file)) for file in files)


from transformers import GPTNeoXConfig


def get_model_and_tokenizer(model_name, app_dir):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    is_valid_lm_head = hasattr(config, "is_valid_lm_head") and config.is_valid_lm_head
    is_decoder = hasattr(config, "is_decoder") and config.is_decoder
    is_encoder_decoder = hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder
    is_valid_question_answering_head = hasattr(config,
                                               "is_valid_question_answering_head") and config.is_valid_question_answering_head

    if is_valid_lm_head and is_decoder:
        model_class = AutoModelForCausalLM
    elif is_valid_lm_head and not is_decoder:
        model_class = AutoModelForMaskedLM
    elif is_encoder_decoder:
        model_class = AutoModelForSeq2SeqLM
    elif is_valid_question_answering_head:
        model_class = AutoModelForQuestionAnswering
    elif isinstance(config, GPTNeoXConfig):
        model_class = AutoModelForCausalLM
    else:
        raise ValueError("Unsupported model type")

    model = model_class.from_pretrained(model_name)

    return tokenizer, model


def ensure_model(model_name, app_dir):
    if not model_exists(app_dir):
        tokenizer, model = get_model_and_tokenizer(model_name, app_dir)
        tokenizer.save_pretrained(app_dir)
        model.save_pretrained(app_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(app_dir)
        config = AutoConfig.from_pretrained(app_dir)

        if config.is_valid_lm_head and config.is_decoder:
            model_class = AutoModelForCausalLM
        elif config.is_valid_lm_head and not config.is_decoder:
            model_class = AutoModelForMaskedLM
        elif config.is_encoder_decoder:
            model_class = AutoModelForSeq2SeqLM
        elif config.is_valid_question_answering_head:
            model_class = AutoModelForQuestionAnswering
        else:
            raise ValueError("Unsupported model type")

        model = model_class.from_pretrained(app_dir)

    return tokenizer, model


models_path = os.path.join(os.path.abspath(os.getcwd()), '../../../models')
# model_name = 'OpenAssistant/stablelm-7b-sft-v7-epoch-3'
# app_dir = os.path.join(models_path, model_name.replace("/", "-"))
selected_model_name = choose_model(model_names)
app_dir = os.path.join(models_path, selected_model_name.replace("/", "-"))
tokenizer, model = ensure_model(selected_model_name, app_dir)
# tokenizer, model = ensure_model(model_name, app_dir)

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
