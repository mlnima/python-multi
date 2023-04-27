import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import concatenate_videoclips, VideoFileClip
import datetime
import random

# Set custom model path
# model_path_dir = "M:/dev/python-multi/models/ttv/text-to-video-ms-1-7b"
model_path_dir = "M:/dev/python-multi/models/ttv/animov"

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
    {"text": "Captain America takes the lead, motivating the Avengers to face Galactus.", "inference_steps": 10,
     "frames": 30},
    {"text": "Iron Man scans Galactus, trying to understand his intentions.", "inference_steps": 10, "frames": 30},
    {"text": "Thor and Hulk prepare for battle, ready to protect Earth.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus lands on Earth, towering over the landscape.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers approach Galactus to communicate with him.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus speaks, expressing his hunger for Earth's resources.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers attempt to reason with Galactus, offering alternative solutions.", "inference_steps": 10,
     "frames": 30},
    {"text": "Galactus refuses their proposals and declares his intention to consume Earth.", "inference_steps": 10,
     "frames": 30},
    {"text": "Iron Man initiates the first attack, firing a barrage of missiles at Galactus.", "inference_steps": 10,
     "frames": 30},
    {"text": "Galactus deflects the missiles effortlessly, smirking at the futile attack.", "inference_steps": 10,
     "frames": 30},
    {"text": "Thor charges at Galactus, swinging his mighty hammer Mjolnir.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus catches Mjolnir mid-air, stunning both Thor and the Avengers.", "inference_steps": 10,
     "frames": 30},
    {"text": "The Hulk leaps into action, attempting to land a powerful punch on Galactus.", "inference_steps": 10,
     "frames": 30},
    {"text": "Galactus swats Hulk away, demonstrating his immense power.", "inference_steps": 10, "frames": 30},
    {"text": "Black Widow and Hawkeye use their agility to evade Galactus' attacks.", "inference_steps": 10,
     "frames": 30},
    {"text": "Doctor Strange casts a spell, attempting to contain Galactus.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus breaks free from Doctor Strange's spell, unaffected.", "inference_steps": 10, "frames": 30},
    {"text": "Captain Marvel flies in, landing a series of powerful blows on Galactus.", "inference_steps": 10,
     "frames": 30},
    {"text": "Galactus is momentarily stunned by Captain Marvel's attack.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers regroup, strategizing a plan to defeat Galactus.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers launch a coordinated attack, striking Galactussimultaneously from multiple angles.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus, overwhelmed by the coordinated assault, starts to weaken.", "inference_steps": 10, "frames": 30},
    {"text": "Ant-Man and the Wasp use their size-shifting abilities to attack Galactus' weak points.", "inference_steps": 10, "frames": 30},
    {"text": "Scarlet Witch uses her powers to restrain Galactus, creating an opening for the Avengers.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers combine their powers, unleashing a devastating attack on Galactus.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus, weakened and defeated, reconsiders his plan to consume Earth.", "inference_steps": 10, "frames": 30},
    {"text": "The Avengers offer Galactus a chance to redeem himself by helping to save other planets.", "inference_steps": 10, "frames": 30},
    {"text": "Galactus accepts the offer, forming a truce with the Avengers and departing Earth peacefully.", "inference_steps": 10, "frames": 30}
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
    output_video_path = os.path.join('./outputs/', output_filename)

    # Ensure the filename is unique
    while os.path.exists(output_video_path):
        random_number = random.randint(1000, 9999)
        output_filename = f'output_{idx}_{timestamp}_{random_number}.mp4'
        output_video_path = os.path.join('./outputs/', output_filename)

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
