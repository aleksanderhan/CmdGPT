# CmdGPT
python program that uses chatgpt to perform various tasks, like interacting with a unix terminal. It can record audio
by holding down the left `crtl` key, and uses whisper api to do a speech-to-text transformation of the audio, that is
piped back to chatgpt, just like writing the input by keyboard.

## How to run

* Install system dependencies, if you are running ubuntu: `sudo apt install portaudio19-dev`
* Install application dependencies `pip install -r requirements.txt`
* Export environment variable `export OPENAI_API_KEY=<your openai api key>`
* Run the script: `python cmdgpt.py`