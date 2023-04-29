# from transformers import MT5ForConditionalGeneration, MT5Tokenizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("persiannlp/mt5-large-parsinlu-translation_en_fa")

model = AutoModelForSeq2SeqLM.from_pretrained("persiannlp/mt5-large-parsinlu-translation_en_fa")


# model_size = "large"
# model_name = f"persiannlp/mt5-{model_size}-parsinlu-translation_en_fa"
#
# tokenizer = MT5Tokenizer.from_pretrained(model_name)
# model = MT5ForConditionalGeneration.from_pretrained(model_name)

tokenizer.save_pretrained("./models")
model.save_pretrained("./models")


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


run_model("Praise be to Allah, the Cherisher and Sustainer of the worlds;")
run_model(
    "shrouds herself in white and walks penitentially disguised as brotherly love through factories and parliaments; offers help, but desires power;")
run_model("He thanked all fellow bloggers and organizations that showed support.")
run_model(
    "Races are held between April and December at the Veliefendi Hippodrome near Bakerky, 15 km (9 miles) west of Istanbul.")
run_model("I want to pursue PhD in Computer Science about social network,what is the open problem in social networks?")




# tokenizer = AutoTokenizer.from_pretrained("./models")
# model = AutoModel.from_pretrained("./models")