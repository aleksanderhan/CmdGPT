import io
import os
import sys
import sounddevice as sd
import numpy as np
import wavio
import openai
from pynput import keyboard
from pynput.keyboard import Key, Listener
from pydub import AudioSegment

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Recording settings
sample_rate = 44100
channels = 1
duration = None
recording = False
audio_data = []
user_input = ""

def on_press(key):
    global recording
    if key == Key.ctrl_l:
        recording = True

def on_release(key):
    global recording
    if key == Key.ctrl_l:
        recording = False
        return False  # Stop listener

def record_callback(indata, frames, time, status):
    if recording:
        audio_data.append(indata.copy())

def text_input_handler(key):
    global user_input
    if key == Key.enter:
        return False
    elif key == Key.backspace:
        user_input = user_input[:-1]
    elif key == Key.shift or key == Key.shift_r or key == Key.ctrl_l:
        pass
    else:
        try:
            user_input += key.char
        except AttributeError:
            pass

print("Enter your text or press and hold the left Ctrl key to record audio:")

with sd.InputStream(samplerate=sample_rate, channels=channels, callback=record_callback):
    with Listener(on_press=on_press, on_release=on_release) as audio_listener:
        with Listener(on_press=text_input_handler) as text_listener:
            text_listener.join()

    if not recording:
        audio_listener.stop()

if len(audio_data) > 0:
    # Convert recorded audio data to numpy array
    recorded_audio = np.concatenate(audio_data, axis=0)

    # Save the recorded audio as a WAV file in memory
    wav_io = io.BytesIO()
    wavio.write(wav_io, recorded_audio, sample_rate, sampwidth=2)

    # Convert the WAV file to MP3 format
    wav_io.seek(0)
    audio = AudioSegment.from_wav(wav_io)
    audio.export("recorded_audio.mp3", format="mp3")

    # Transcribe the recorded audio using OpenAI API
    audio_file = open("./recorded_audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    user_input = transcript

print("\nUser input: ", user_input)
