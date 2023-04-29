import os

import torch
from abcmidi import ABCMidi
from samplings import top_p_sampling, temperature_sampling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def abc_to_midi(abc_notation, midi_file_path):
    abc_midi = ABCMidi(abc_notation)
    if abc_midi.is_valid():
        abc_midi.write(midi_file_path)
        return midi_file_path
    else:
        print("Error: Unable to convert the ABC notation to MIDI.")
        return None


# Define model path
model_path_dir = "M:/dev/python-multi/models/ttm/text-to-music"

# Create directory if it doesn't exist
if not os.path.exists(model_path_dir):
    os.makedirs(model_path_dir)

# Define model name
model_name = 'sander-wood/text-to-music'
model_path = os.path.join(model_path_dir, model_name)

# Load or download the model and tokenizer
if not os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Define variables for generation
max_length = 1024
top_p = 0.9
temperature = 1.0

# Input text
text = "This is a traditional Irish dance music."

# Tokenize input text
input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)['input_ids']

# Initialize decoder input
decoder_start_token_id = model.config.decoder_start_token_id
eos_token_id = model.config.eos_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])

# Generate ABC notation
for t_idx in range(max_length):
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    probs = outputs.logits[0][-1]
    probs = torch.nn.Softmax(dim=-1)(probs).detach().numpy()
    sampled_id = temperature_sampling(probs=top_p_sampling(probs, top_p=top_p, return_probs=True),
                                      temperature=temperature)
    decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)

    if sampled_id != eos_token_id:
        continue
    else:
        tune = "X:1\n"
        tune += tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        print("Generated ABC notation:\n", tune)
        break

# Define file paths
midi_file_path = 'output.mid'

# Convert ABC notation to MIDI
abc_to_midi(tune, midi_file_path)
