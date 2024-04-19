import torch
import torchaudio
from transformers import ClapProcessor, ClapModel

class Interrogator:
    def __init__(self, model_name="laion/clap-htsat-unfused"):
        # Load the processor and model
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load tags
        self.tags = self.load_tags()

    def load_tags(self, file_path='tags.json'):
        import json
        with open(file_path, 'r') as file:
            tags_data = json.load(file)
            tags = sum(tags_data.values(), [])
        return list(set(tags))

    def tag(self, audio_input, sr=None, top_n=10):
        # Check if audio_input is a string path, if so, load the audio
        if isinstance(audio_input, str):
            audio, sr = torchaudio.load(audio_input)
            audio = audio.squeeze(0) 
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input
            if sr is None:
                raise ValueError("Sampling rate must be provided with audio tensor.")
        else:
            raise TypeError("Invalid input type for audio_input. Must be a file path or torch.Tensor.")

        # Process inputs
        inputs = self.processor(text=self.tags, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
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