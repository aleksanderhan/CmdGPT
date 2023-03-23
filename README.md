# CmdGPT
A Python program that uses chatgpt to perform various tasks, like interacting with a unix terminal. It can record audio
by holding down the right `ctrl` key, and uses whisper api to do a speech-to-text transcription of the audio, that is
then piped back to chatgpt.

## Sample interaction
>> launch a calculator

<cmd>gnome-calculator &</cmd>

I'm sorry, I couldn't launch the calculator. It seems that 'gnome-calculator' is not installed on this system. Would you like me to try launching another calculator?

>> install gnome-calculator

What os do you use?

>> ubuntu

<cmd>sudo apt install gnome-calculator</cmd>

I have installed 'gnome-calculator' successfully.

<cmd>gnome-calculator &</cmd>

I have launched the calculator.


## How to run

* Install system dependencies, if you are running ubuntu: `sudo apt install portaudio19-dev`
* Install application dependencies `pip install -r requirements.txt`
* Export environment variable `export OPENAI_API_KEY=<your openai api key>`
* Run the script: `python cmdgpt.py`
