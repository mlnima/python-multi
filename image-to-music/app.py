import gradio as gr
import openai
import numpy as np
import time
import base64
import ffmpeg
from sentence_transformers import SentenceTransformer
from audio2numpy import open_audio
import httpx
import json
import os
import requests
import urllib
import pydub
from os import path
from pydub import AudioSegment
import re

MUBERT_LICENSE = os.environ.get('MUBERT_LICENSE')
MUBERT_TOKEN = os.environ.get('MUBERT_TOKEN')

# img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
img_to_text = gr.Blocks.load(name="spaces/fffiloni/CLIP-Interrogator-2")

from share_btn import community_icon_html, loading_icon_html, share_js
from utils import get_tags_for_prompts, get_mubert_tags_embeddings

minilm = SentenceTransformer('all-MiniLM-L6-v2')
mubert_tags_embeddings = get_mubert_tags_embeddings(minilm)

MUBERT_LICENSE = os.environ.get('MUBERT_LICENSE')
MUBERT_TOKEN = os.environ.get('MUBERT_TOKEN')


def get_pat_token():
    r = httpx.post('https://api-b2b.mubert.com/v2/GetServiceAccess',
                   json={
                       "method": "GetServiceAccess",
                       "params": {
                           "email": "mail@mail.com",
                           "phone": "+11234567890",
                           "license": MUBERT_LICENSE,
                           "token": MUBERT_TOKEN,

                       }
                   })

    rdata = json.loads(r.text)
    assert rdata['status'] == 1, "probably incorrect e-mail"
    pat = rdata['data']['pat']
    # print(f"pat: {pat}")
    return pat


def get_music(pat, prompt, track_duration, gen_intensity, gen_mode):
    if len(prompt) > 200:
        prompt = prompt[:200]

    r = httpx.post('https://api-b2b.mubert.com/v2/TTMRecordTrack',
                   json={
                       "method": "TTMRecordTrack",
                       "params":
                           {
                               "text": prompt,
                               "pat": pat,
                               "mode": gen_mode,
                               "duration": track_duration,
                               "intensity": gen_intensity,
                               "format": "wav"
                           }
                   })

    rdata = json.loads(r.text)

    # print(f"rdata: {rdata}")
    assert rdata['status'] == 1, rdata['error']['text']
    track = rdata['data']['tasks'][0]['download_link']
    print(track)

    local_file_path = "sample.wav"

    # Download the MP3 file from the URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:93.0) Gecko/20100101 Firefox/93.0'}

    retries = 3
    delay = 5  # in seconds
    while retries > 0:
        response = requests.get(track, headers=headers)
        if response.status_code == 200:
            break
        retries -= 1
        time.sleep(delay)
    response = requests.get(track, headers=headers)
    print(f"{response}")
    # Save the downloaded content to a local file
    with open(local_file_path, 'wb') as f:
        f.write(response.content)
        return "sample.wav", track


def get_results(text_prompt, track_duration, gen_intensity, gen_mode):
    pat_token = get_pat_token()
    music = get_music(pat_token, text_prompt, track_duration, gen_intensity, gen_mode)
    return pat_token, music[0], music[1]


def get_prompts(uploaded_image, track_duration, gen_intensity, gen_mode, openai_api_key):
    print("calling clip interrogator")
    # prompt = img_to_text(uploaded_image, "ViT-L (best for Stable Diffusion 1.*)", "fast", fn_index=1)[0]

    prompt = img_to_text(uploaded_image, 'best', 4, fn_index=1)[0]
    print(prompt)
    clean_prompt = clean_text(prompt)
    print(f"prompt cleaned: {clean_prompt}")
    musical_prompt = 'You did not use any OpenAI API key to pimp your result :)'
    if openai_api_key is not None:
        gpt_adaptation = try_api(prompt, openai_api_key)
        if gpt_adaptation[0] != "oups":
            musical_prompt = gpt_adaptation[0]
            print(f"musical adapt: {musical_prompt}")
            music_result = get_results(musical_prompt, track_duration, gen_intensity, gen_mode)
        else:
            music_result = get_results(clean_prompt, track_duration, gen_intensity, gen_mode)
    else:
        music_result = get_results(clean_prompt, track_duration, gen_intensity, gen_mode)

    show_prompts = f"""
        CLIP Interrogator Caption: '{prompt}'
        —
        OpenAI Musical Adaptation: '{musical_prompt}'
        —
        Audio file link: {music_result[2]}
    """
    # wave_file = convert_mp3_to_wav(music_result[1])

    time.sleep(1)
    return gr.Textbox.update(value=show_prompts, visible=True), music_result[1], gr.update(visible=True), gr.update(
        visible=True), gr.update(visible=True)


def try_api(message, openai_api_key):
    try:
        response = call_api(message, openai_api_key)
        return response, "<span class='openai_clear'>no error</span>"
    except openai.error.Timeout as e:
        # Handle timeout error, e.g. retry or log
        # print(f"OpenAI API request timed out: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request timed out: <br />{e}</span>"
    except openai.error.APIError as e:
        # Handle API error, e.g. retry or log
        # print(f"OpenAI API returned an API Error: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API returned an API Error: <br />{e}</span>"
    except openai.error.APIConnectionError as e:
        # Handle connection error, e.g. check network or log
        # print(f"OpenAI API request failed to connect: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request failed to connect: <br />{e}</span>"
    except openai.error.InvalidRequestError as e:
        # Handle invalid request error, e.g. validate parameters or log
        # print(f"OpenAI API request was invalid: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request was invalid: <br />{e}</span>"
    except openai.error.AuthenticationError as e:
        # Handle authentication error, e.g. check credentials or log
        # print(f"OpenAI API request was not authorized: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request was not authorized: <br />{e}</span>"
    except openai.error.PermissionError as e:
        # Handle permission error, e.g. check scope or log
        # print(f"OpenAI API request was not permitted: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request was not permitted: <br />{e}</span>"
    except openai.error.RateLimitError as e:
        # Handle rate limit error, e.g. wait or log
        # print(f"OpenAI API request exceeded rate limit: {e}")
        return "oups", f"<span class='openai_error'>OpenAI API request exceeded rate limit: <br />{e}</span>"


def call_api(message, openai_api_key):
    instruction = "Convert in less than 200 characters this image caption to a very concise musical description with musical terms, as if you wanted to describe a musical ambiance, stricly in English"

    print("starting open ai")
    augmented_prompt = f"{instruction}: '{message}'."
    openai.api_key = openai_api_key

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=augmented_prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )

    # print(response)

    # return str(response.choices[0].text).split("\n",2)[2]
    return str(response.choices[0].text).lstrip('\n')


def get_track_by_tags(tags, pat, duration, gen_intensity, gen_mode, maxit=20):
    r = httpx.post('https://api-b2b.mubert.com/v2/RecordTrackTTM',
                   json={
                       "method": "RecordTrackTTM",
                       "params": {
                           "pat": pat,
                           "duration": duration,
                           "format": "wav",
                           "intensity": gen_intensity,
                           "tags": tags,
                           "mode": gen_mode
                       }
                   })

    rdata = json.loads(r.text)
    print(rdata)
    # assert rdata['status'] == 1, rdata['error']['text']
    trackurl = rdata['data']['tasks'][0]

    print('Generating track ', end='')
    for i in range(maxit):
        r = httpx.get(trackurl)
        if r.status_code == 200:
            return trackurl
        time.sleep(1)


def generate_track_by_prompt(pat, prompt, duration, gen_intensity, gen_mode):
    try:
        _, tags = get_tags_for_prompts(minilm, mubert_tags_embeddings, prompt)[0]
        result = get_track_by_tags(tags, pat, int(duration), gen_intensity, gen_mode)
        print(result)
        return result, ",".join(tags), "Success"
    except Exception as e:
        return None, "", str(e)


def convert_mp3_to_wav(mp3_filepath):
    wave_file = "file.wav"

    sound = AudioSegment.from_mp3(mp3_filepath)
    sound.export(wave_file, format="wav")

    return wave_file


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_nonalphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def clean_text(text):
    clean_text = remove_nonalphanumeric(text)
    clean_text = remove_emoji(clean_text)
    clean_text = re.sub(r'\d+', '', clean_text)  # Remove any number
    return clean_text


article = """

    <div class="footer">
        <p>

        Follow <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a> for future updates 🤗
        </p>
    </div>

    <div id="may-like-container" style="display: flex;justify-content: center;flex-direction: column;align-items: center;margin-bottom: 30px;">
        <p style="font-size: 0.8em;margin-bottom: 4px;">You may also like: </p>
        <div id="may-like" style="display: flex;flex-wrap: wrap;align-items: center;height: 20px;">
            <svg height="20" width="122" style="margin-left:4px;margin-bottom: 6px;">       
                 <a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music" target="_blank">
                    <image href="https://img.shields.io/badge/🤗 Spaces-Riffusion-blue" src="https://img.shields.io/badge/🤗 Spaces-Riffusion-blue.png" height="20"/>
                 </a>
            </svg>
        </div>
    </div>

"""

with gr.Blocks(css="style.css") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""<div style="text-align: center; max-width: 700px; margin: 0 auto;">
                <div
                style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.8rem;
                    font-size: 1.75rem;
                "
                >
                <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
                    Image to Music
                </h1>
                </div>
                <p style="margin-bottom: 10px; font-size: 94%">
                Sends an image in to <a href="https://huggingface.co/spaces/pharma/CLIP-Interrogator" target="_blank">CLIP Interrogator</a>
                to generate a text prompt which is then run through 
                <a href="https://huggingface.co/Mubert" target="_blank">Mubert</a> text-to-music to generate music from the input image!
                </p>
            </div>""")

        input_img = gr.Image(type="filepath", elem_id="input-img")
        prompts_out = gr.Textbox(label="Text Captions", visible=False, elem_id="prompts_out",
                                 info="If player do not work, try to copy/paste the link in a new browser window")
        music_output = gr.Audio(label="Result", type="filepath", elem_id="music-output").style(height="5rem")
        # music_url = gr.Textbox(max_lines=1, info="If player do not work, try to copy/paste the link in a new browser window")
        # text_status = gr.Textbox(label="status")
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

        with gr.Accordion(label="Music Generation Options", open=False):
            openai_api_key = gr.Textbox(type="password", label="🔐 Your OpenAI API Key (optional)",
                                        placeholder="sk-123abc...",
                                        info="You can use your OpenAI key to adapt CLIP Interrogator caption to a musical translation.")
            track_duration = gr.Slider(minimum=20, maximum=120, value=55, ustep=5, label="Track duration",
                                       elem_id="duration-inp")
            with gr.Row():
                gen_intensity = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Intensity")
                gen_mode = gr.Radio(label="mode", choices=["track", "loop"], value="loop")

        generate = gr.Button("Generate Music from Image")

        gr.HTML(article)

    generate.click(get_prompts, inputs=[input_img, track_duration, gen_intensity, gen_mode, openai_api_key],
                   outputs=[prompts_out, music_output, share_button, community_icon, loading_icon], api_name="i2m")
    share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=32).launch()
