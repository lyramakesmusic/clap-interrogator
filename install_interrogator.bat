@echo off

python -m venv ./venv
call venv/Scripts/activate

pip install laion-clap
pip install -U transformers==4.30.0
pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
