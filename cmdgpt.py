import tiktoken
import openai
import pprint
import subprocess
import os
import io
import sounddevice as sd
import numpy as np
import wavio
import asyncio
import backoff
from bs4 import BeautifulSoup
from pynput import keyboard
from pynput.keyboard import Key, Listener
from pydub import AudioSegment
from aioconsole import ainput

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except:
    print("You need to set the OPENAI_API_KEY environment variable to run this script.")
    exit()


models = [
    {
        "model": "gpt-3.5-turbo",
        "max_context": 4096
    },
    {
        "model": "gpt-4",
        "max_context": 8192
    }
]

directives = [
    [{"role": "system", "content": "You are a personal assistant that answers questions as best as you can."}],
    [{
        "role": "system", 
        "content": """You are a python code generator that translates natural language input from the user into python code. When the user says something, you respond in 
python code and nothing else. The user can ask you to refactor the previous code, and you will reply with the refactored code."""
    },
    {
        "role": "system",
        "content": "The following is an example:"
    },
    {
        "role": "system",
        "content": "user: I want a function that calculates the fibonacci sequence."
    },
    {
        "role": "system",
        "content": """assistant:
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)"""
    },
    {
        "role": "system",
        "content": "user: I want a faster approach."
    },
    {
        "role": "system",
        "content": """assistant:
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
    }],
    [{
        "role": "system",
        "content": """You are a natural language to unix terminal computer interface program, that takes natural language description of tasks that can be performed on the unix terminal,
and emits a unix command on the form '<cmd>unix command</cmd>' to be ran in the terminal. You will get a response from the system of the output of the command.
You take this output and return it to the user in a natural language format. ALWAYS TELL THE USER WHAT THE RESULT OF THE COMMAND WAS. You can ask the user to be more 
spesific or explain himself before you run a command if something is unclear. 
WHAT IS IMPORTANT IS THAT WHEN YOU WANT TO RUN A COMMAND, YOU HAVE TO ENCLOSE THE COMMAND IN TAGS LIKE THIS: '<cmd>command</cmd>'. NEVER USE THE CMD TAG WHEN YOU DON'T
WANT TO RUN A COMMAND."""
    },
    {
        "role": "system",
        "content": "The following is an example:"
    },
    {
        "role": "system",
        "content": "user: What are the files in the directory?"
    },
    {
        "role": "system",
        "content": "assistant: Could you specify which directory?"
    },
    {
        "role": "system",
        "content": "user: The current working directory."
    }, 
    {
        "role": "system",
        "content": "assistant: <cmd>ls</cmd>"
    },
    {
        "role": "system",
        "content": "system: file1.txt videos"
    },
    {
        "role": "system",
        "content": "assistant: The content of the current working directory is: file1.txt and videos."
    },
    {
        "role": "system",
        "content": "user: Which are files and which are directories?"
    },
    {
        "role": "system",
        "content": "assistant: <cmd>ls -F<cmd>"
    },
    {
        "role": "system",
        "content": "system: file.txt videos/"
    },
    {
        "role": "system",
        "content": "assistant: 'file.txt' is a file, while 'videos' is a directory."
    },
    {
        "role": "system",
        "content": "user: whats the content of the file?"
    },
    {
        "role": "system",
        "content": "assistant: <cmd>cat file1.txt</cmd>"
    },
    {
        "role": "system",
        "content": "system: Lorem Ipsum"
    },
    {
        "role": "system",
        "content": "assistant: The content of file1.txt is:\n'Lorem Ipsum'"
    },
    {
        "role": "system",
        "content": "user: start firefox."
    }, 
    {
        "role": "system",
        "content": "assistant: <cmd>firefox &</cmd>"
    }]
]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_response(messages, n=1, stream=True, temp=1, model="gpt-3.5-turbo", max_context_length=4096):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
        n=n,
        max_tokens=max_context_length-num_tokens_from_messages(messages),
        stream = stream
    )
    
    return response


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-4":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant

        return num_tokens
    else:
        raise NotImplementedError()


def has_tag(content, tag='cmd'):
    soup = BeautifulSoup(content, "lxml")
    return soup.find(tag)


def summarize_chat(text, model, max_context_length):
    messages = [
        {   
            "role": "system",
            "content": """You are a text bot that summarizes and compresses text. You receive a text of the history of a chat between three participants, user, assistant and system, 
on the form: 
-----
<msg1>\{participant 1\}: \{some text written by participant 1\}</msg1>
<msg2>\{participant 2\}: \{some text written by participant 2\}</msg2>
...
-----
Your job is to summarize the conversation between the participants and compress it down to the essentials to be used as a reference later."""
        },
        {
            "role": "user",
            "content": "Summarize and compress the following chat, but keep the details:\n-----\n{}\n-----\n".format(text)
        }
    ]
    response = generate_response(messages, stream=False, model=model["model"], max_context_length=model["max_context"])
    return response['choices'][0]['message']['content']


def compress_text(text, model, max_context_length):
    messages = [
        {   
            "role": "system",
            "content": """You are a text bot that recieves text that is too long to fit into a message. Your job is to compress it by being more terse, removing unnecessary
words and sentences or rewriting parts of it, but still keeping the information intact in the message. The whole point is to compress the text into fewer tokens."""
        },
        {
            "role": "user",
            "content": "Summarize and compress the following text, but keep the details:\n-----\n{}\n-----\n".format(text)
        }
    ]
    response = generate_response(messages, stream=False, model=model["model"], max_context_length=model["max_context"])
    return response['choices'][0]['message']['content']



def join_messages(messages):
    res = ""
    for i, msg in enumerate(messages):
        m = f"<msg{i}>" + msg['role'] + ": " + msg['content'] + f"</msg{i}>\n"
        res += m
    return res


class CmdGPT:

    def __init__(self):
        self.directive_number = 0
        self.directive = directives[self.directive_number]
        self.messages = []
        self.directive_length = num_tokens_from_messages(self.directive)
        self.model_number = 0
        self.model = models[self.model_number]
        self.temperature = 1

        self.recording = False
        self.sample_rate = 44100
        self.audio_filename = "audio_input.mp3"

    def add_message(self, message):
        tokens_in_message = num_tokens_from_messages([message])
        if tokens_in_message > self.model["max_context"]:
            print(message)
            print("THE MESSAGE IS TOO BIG ({} TOKENS) TO FIT IN THE CHAT, SKIPPING MESSAGE\n".format(tokens_in_message))
        elif tokens_in_message > int(self.model["max_context"]*0.8):
            print("THE MESSAGE IS BIG, TRYING TO COMPRESS IT FIRST")
            compressed = compress_text(message["content"], self.model["model"], self.model["max_context"])
            message = {"role": message['role'], "content": compressed['content']}
        self.messages.append(message)

    def print_help(self):
        print("This chatbot has a few reserved keywords for system management.")
        print("0. exit - Exit program.")
        print("1. help - Print help message")
        print("2. msg - Print message log.")
        print("3. change - Print an enumerated list of different directive choices.")
        print("4. compress - Compress message log")
        print("5. model - Change model")
        print("6. clear - Clear the message log.")
        print("7. temp - Change temperature.")

    def run(self):
        self.print_help()

        while True:
            print()
            print("-------------------------------------------------------------------------------------------")

            self.audio_data = []
            self.user_input = None
            asyncio.run(self.get_user_input())
            print()

            if self.user_input == "":
                continue
            elif self.user_input == "exit" or self.user_input == "!0":
                exit()
            elif self.user_input == "help" or self.user_input == "!1":
                self.print_help()
            elif self.user_input == "msg" or self.user_input == "!2":
                for msg in self.messages:
                    pprint.pprint(msg)
            elif self.user_input == "change" or self.user_input == "!3":
                asyncio.run(self.change_directive())
            elif self.user_input == "compress" or self.user_input == "!4":
                self.compress_and_clear_messages()
            elif self.user_input == "model" or self.user_input == "!5":
                asyncio.run(self.change_model())
            elif self.user_input == "clear" or self.user_input == "!6":
                self.messages = []
            elif self.user_input == "temp" or self.user_input == "!7":
                asyncio.run(self.change_temperature())
            else:
                self.handle_input({"role": "user", "content": self.user_input, "compressed": False})

    def handle_input(self, msg):
        self.add_message(msg)

        if num_tokens_from_messages(self.messages) > int(self.model["max_context"]*0.7):
            self.compress_and_clear_messages()                  

        content = self.generate_response(self.directive + self.messages)
        self.add_message({"role": "assistant", "content": content, "compressed": False})
        cmd = has_tag(content, tag="cmd")

        if self.directive_number == 2 and cmd is not None: # CmdGPT
            result = self.run_cmd(cmd.string)
            self.handle_input({"role": "system", "content": result, "compressed": False})
            
        print()

    async def get_user_input(self):
        audio_task = asyncio.create_task(self.audio_input())
        text_task = asyncio.create_task(self.text_input())
        done, pending = await asyncio.wait({audio_task, text_task}, return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending:
            task.cancel()

        if len(self.audio_data) > 0:
            # Convert recorded audio data to numpy array
            recorded_audio = np.concatenate(self.audio_data, axis=0)

            # Save the recorded audio as a WAV file in memory
            wav_io = io.BytesIO()
            wavio.write(wav_io, recorded_audio, self.sample_rate, sampwidth=2)

            # Convert the WAV file to MP3 format
            wav_io.seek(0)
            audio = AudioSegment.from_wav(wav_io)
            audio.export(self.audio_filename, format="mp3")

            # Transcribe the recorded audio using OpenAI API
            with open(self.audio_filename, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
                print(transcript["text"])
                self.user_input = transcript["text"]

    async def audio_input(self):
        loop = asyncio.get_event_loop()
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.record_callback):
            with Listener(on_press=self.on_press, on_release=self.on_release) as audio_listener:
                await loop.run_in_executor(None, audio_listener.join)

    async def text_input(self):
        self.user_input = await ainput("{}, directive: {}, user_input ({}/{})>> ".format(
            self.model["model"], 
            self.directive_number, 
            num_tokens_from_messages(self.messages), 
            self.model["max_context"] - self.directive_length
        ))
        print()

    def on_press(self, key):
        if key == Key.ctrl_r:
            self.recording = True

    def on_release(self, key):
        if key == Key.ctrl_r:
            self.recording = False
            return False  # Stop listener

    def record_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    async def change_model(self):
        user_input = None
        while not user_input and not user_input == 0:
            print("Choose model:")
            for i, model in enumerate(models):
                print(i, "-", model["model"])
            try:
                user_input = int(await ainput())
                self.model_number = user_input
                self.model = models[self.model_number]
            except Exception as e:
                print(e)
                print("Choose a number")


    async def change_directive(self):
        user_input = None
        while not user_input and not user_input == 0:
            print("Choose directive:")
            for i, directive in enumerate(directives):
                print(i, "-", directive[0]['content'])
                print()
            try:
                user_input = int(await ainput())
                self.directive_number = user_input
                self.directive = directives[self.directive_number]
                self.directive_length = num_tokens_from_messages(self.directive)
            except Exception as e:
                print(e)
                print("Choose a number")

    async def change_temperature(self):
        user_input = None
        while not user_input and not user_input == 0:
            print("Choose a temperature in the range [0-2]:")
            try:
                user_input = float(await ainput())
                self.temperature = user_input
            except Exception as e:
                print(e)
                print("Choose a number between 0 and 2.")

    def run_cmd(self, cmd):
        print()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            o, e = proc.communicate()
            return o.decode("utf-8") + e.decode("utf-8")    
        except Exception as e:
            print(e)

    def compress_and_clear_messages(self):
        to_compress = []
        while num_tokens_from_messages(self.messages) > int(self.model["max_context"]*0.3):
            if not self.messages[0]['compressed']:
                to_compress.append(self.messages[0])
            self.messages.pop(0)

        joined_text = join_messages(to_compress)
        summary = summarize_chat(joined_text, self.model["model"], self.model["max_context"])
        self.messages.insert(0, {
                "role": "system",
                "content": "This is a summary of the previous conversation:\n" + summary,
                "compressed": True
            })  

    def generate_response(self, messages):
        content = ""
        for resp in generate_response(messages, temp=self.temperature, model=self.model["model"], max_context_length=self.model["max_context"]):
            try:
                delta = resp["choices"][0]["delta"]["content"]
                content = content + delta
                print(delta, end="")
            except KeyError as ke:
                pass
        return content

    def __del__(self):
        if os.path.isfile(self.audio_filename):
            os.remove(self.audio_filename)


qti = CmdGPT()
try:
    qti.run()
except KeyboardInterrupt as ki:
    print()
    exit()
