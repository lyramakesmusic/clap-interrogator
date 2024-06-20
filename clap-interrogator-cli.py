import argparse
from pathlib import Path
from clap_interrogator import Interrogator

def generate_captions(input_path):
    tags_path = Path(__file__).resolve().parent / "tags.json"
    interrogator = Interrogator(tags=str(tags_path))
    
    def process_wav_file(file):
        tags = interrogator.tag(str(file))
        out_file = file.with_suffix('.txt')
        out_file.write_text(', '.join(tags))
        print(f"Generated captions for {file}")

    input_path = Path(input_path)

    if input_path.is_dir():
        wav_files = list(input_path.glob('*.wav'))
        if not wav_files:
            print(f"No WAV files found in {input_path}")
            return
        for file in wav_files:
            process_wav_file(file)
        
    elif input_path.is_file() and input_path.suffix == '.wav':
        process_wav_file(input_path)

    else:
        print("Invalid input path. Provide a folder containing WAV files or a single WAV file.")
        return

def main():
    parser = argparse.ArgumentParser(description="Generate captions for audio files using the CLAP model.")
    parser.add_argument('--input-path', type=str, required=True, help='Path to a folder containing WAV files or a single WAV file.')
    args = parser.parse_args()

    input_path = args.input_path
    generate_captions(input_path)

if __name__ == "__main__":
    main()