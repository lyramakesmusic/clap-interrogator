from setuptools import setup, find_packages

setup(
    name="clap_interrogator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.0",
        "librosa>=0.9.2"
    ],
)
