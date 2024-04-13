from transformers import ClapProcessor, ClapModel
import torch
import librosa
import os
import json
import random
import time
from tqdm import tqdm

model_name = "laion/clap-htsat-unfused"

# Load the processor and model
processor = ClapProcessor.from_pretrained(model_name)
model = ClapModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f'loaded {model_name}')

# Load tags from JSON file
with open('tags.json', 'r') as file:
    tags_data = json.load(file)
    tags = sum(tags_data.values(), [])

tags = list(set(tags))

def interrogate(audio_file, top_n=10):
    global model, processor, tags

    # process inputs
    audio, sr = librosa.load(audio_file, sr=48000)
    audio_tensor = torch.tensor(audio, device=device)
    inputs = processor(text=tags, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # compute similarity
    outputs = model(**inputs)
    logits_per_audio = outputs.logits_per_audio
    probs = logits_per_audio.softmax(dim=-1)

    # Get the top top_n=10 indices for each audio
    top_probs, top_indices = probs.topk(top_n, dim=1)
    top_matches = [tags[i] for i in top_indices[0].tolist()]

    return top_matches

output_file = "interrogator_data.jsonl"
audio_paths = []
audio_directory = "."

# Collect audio file paths
for root, dirs, files in os.walk(audio_directory):
    for audio_file in files:
        if "__MACOSX" in os.path.join(root, audio_file):
            continue
        if audio_file.endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff')):
            audio_paths.append(os.path.join(root, audio_file)[2:])

random.shuffle(audio_paths)

# Process each audio file
with open(output_file, "w") as f:

    for audio_path in tqdm(audio_paths):

        try:
            t0 = time.time()
            top_matches = interrogate(audio_path, top_n=15)

            line = {"path": audio_path.replace("\\", "/"), "caption": ', '.join(top_matches)}
            f.write(json.dumps(line) + "\n")
        except Exception as e:
            import traceback
            with open('log.txt', 'a') as log:
                traceback.print_exc(file=log)

        # print(f" - ({round(time.time() - t0, 3)}s) {audio_path}: {', '.join(top_matches)}")
