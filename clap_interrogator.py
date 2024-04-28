import librosa
import torch
import torchaudio
from transformers import ClapProcessor, ClapModel

class Interrogator:
    def __init__(self, model_name="laion/clap-htsat-unfused", tags="clap-interrogator/tags.json"):

        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if isinstance(tags, str):
            self.tags = self.load_tags(tags)
        elif isinstance(tags, list):
            self.tags = list(set(tags))
        else:
            raise ValueError("Tags must be a file path (str) or a list of tags (list).")

    def load_tags(self, file_path='tags.json'):
        import json
        with open(file_path, 'r') as file:
            tags_data = json.load(file)
            tags = sum(tags_data.values(), [])
        return list(set(tags))

    def tag(self, audio_input, sr=None, top_n=10):

        # Audio loading and reshaping
        if isinstance(audio_input, str):
            audio, sr = librosa.load(audio_input, sr=48000)
            audio_tensor = torch.tensor(audio, device=self.device)

        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input
            if sr is None:
                raise ValueError("Sampling rate must be provided with audio tensor.")
        else:
            raise TypeError("Invalid input type for audio_input. Must be a filepath or torch.Tensor.")

        # Process inputs
        inputs = self.processor(text=self.tags, audios=[audio], sampling_rate=48000, return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Compute similarity
        with torch.no_grad(): 
            outputs = self.model(**inputs)
        logits_per_audio = outputs.logits_per_audio
        probs = logits_per_audio.softmax(dim=-1)
        
        # Get the top tags
        top_probs, top_indices = probs.topk(top_n, dim=1)
        top_matches = [self.tags[i] for i in top_indices[0].tolist()]
        
        return top_matches
