# CLAP Interrogator

A simple Python class for audio tagging using the CLAP model from the LAION suite

## Installation

Clone and install:

```
git clone https://github.com/lyramakesmusic/interrogator.git
cd interrogator
pip install -e .
```

## Usage

```py
from clap_interrogator import Interrogator

interrogator = Interrogator()

# Tag audio by providing file path:
tags = interrogator.tag('/path/to/audio/file.wav')

# Tag an already loaded audio tensor:
import torchaudio
audio_tensor, sr = torchaudio.load('/path/to/audio/file.wav')
tags = interrogator.tag(audio_tensor, sr)

print(tags) # ["tag 1", "tag 2", "tag 3" . . .]

# I want more tags!
tags = interrogator.tag('/path/to/audio/file.wav', top_n=50)

# Load a different CLAP model:
interrogator = Interrogator(model_name="laion/clap-htsat-unfused")

# Use different tags to interrogate against:
interrogator = Interrogator(tags="more_tags.json")
interrogator = Interrogator(tags=["Bright", "Mellow", "Strings"])
```

Any `tags.json` file you provide is expected to follow this format:

```json
{
  "category": [
    "tag1", "tag2", "tag3"
  ],
  "category2": [
    "tag4", "tag5", "tag6", "tag7"
  ]
}
```

Development version (with WIP genetic algo) here: https://colab.research.google.com/drive/1GoLrpSbwm9Bersz3n42F2jVvAXr02hMB
