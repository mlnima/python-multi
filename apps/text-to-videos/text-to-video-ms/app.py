import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import concatenate_videoclips, VideoFileClip
import datetime
import random

models_path = os.path.join(os.path.abspath(os.getcwd()), '../../../models')

# Set custom model path
# model_path_dir = os.path.join(models_path,"text-to-video-ms-1-7b")
model_path_dir = os.path.join(models_path, "animov-0.1.1")

if not os.path.exists(model_path_dir):
    os.makedirs(model_path_dir)

# Check if the model exists and download if needed
model_name = "strangeman3107/animov-0.1.1"
model_path = os.path.join(model_path_dir, model_name)

if not os.path.exists(model_path):
    DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").save_pretrained(model_path)

# Load the model from custom path
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Advanced prompts with custom settings
prompts = [
    {"text": "Galactus approaches Earth, causing panic among the citizens.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers assemble to discuss the Galactus situation.", "inference_steps": 10, "frames": 30},
]

video_files = []

# Loop through prompts and generate videos
for idx, prompt in enumerate(prompts):
    video_frames = pipe(prompt["text"], num_inference_steps=prompt["inference_steps"],
                        num_frames=prompt["frames"]).frames

    # Generate unique output filename with date and random number
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_number = random.randint(1000, 9999)
    output_filename = f'output_{idx}_{timestamp}_{random_number}.mp4'
    output_video_path = os.path.join('outputs/', output_filename)

    # Ensure the filename is unique
    while os.path.exists(output_video_path):
        random_number = random.randint(1000, 9999)
        output_filename = f'output_{idx}_{timestamp}_{random_number}.mp4'
        output_video_path = os.path.join('outputs/', output_filename)

    export_to_video(video_frames, output_video_path=output_video_path)
    video_files.append(output_video_path)

# Load video clips
video_clips = [VideoFileClip(video_file) for video_file in video_files]

# Concatenate video clips
concatenated_clip = concatenate_videoclips(video_clips)

# Write the concatenated video to a single file
concatenated_clip.write_videofile('concatenated_video.mp4')

# Close video clips to release resources
for clip in video_clips:
    clip.close()
