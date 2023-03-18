import tiktoken
import openai
import pprint
import subprocess
import os
from bs4 import BeautifulSoup


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
	[{"role": "system", "content": "You are a python code generator that translates natural language input from the user into python code. When the user says something, you respond in python code and nothing else."}],
	[{
		"role": "system",
		"content": """You are a natural language to unix terminal computer interface program, that takes natural language description of tasks that can be performed on the unix terminal,
					and emits a unix command on the form '<cmd>unix command</cmd>' to be ran in the terminal. You will get a response from the system of the output of the command.
					You take this output and return it to the user in a natural language format. ALWAYS TELL THE USER WHAT THE RESULT OF THE COMMAND WAS. You can ask the user to be more 
					spesific or explain himself before you run a command if something is unclear. 
					WHAT IS IMPORTANT IS THAT WHEN YOU WANT TO RUN A COMMAND, YOU HAVE TO ENCLOSE THE COMMAND IN TAGS LIKE THIS: '<cmd>command</cmd>'. NEVER USE THE CMD TAG WHEN YOU DON'T
					WANT TO RUN A COMMAND. Here's an example:"""
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
		"content": "system: file1.txt"
	},
	{
		"role": "system",
		"content": "assistant: The content of the current working directory is: 'file1.txt'."
	},
	{
		"role": "system",
		"content": "user: whats the content of that file?"
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
		"content": "assistant: <cmd>firefox</cmd>"
	}]
]


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
	response = generate_response(messages, stream=False, model=self.model["model"], max_context_length=self.model["max_context"])
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
	response = generate_response(messages, stream=False, model=self.model["model"], max_context_length=self.model["max_context"])
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

	def run(self):
		self.print_help()

		while True:
			print()
			print("-------------------------------------------------------------------------------------------")
			print("{}, directive: {}, user_input ({}/{})>> ".format(self.model["model"], self.directive_number, num_tokens_from_messages(self.messages), self.model["max_context"] - self.directive_length), end="")
			user_input = input()
			print()

			if user_input == "":
				continue
			elif user_input == "exit" or user_input == "!0":
				exit()
			elif user_input == "help" or user_input == "!1":
				self.print_help()
			elif user_input == "msg" or user_input == "!2":
				for msg in self.messages:
				    pprint.pprint(msg)
			elif user_input == "change" or user_input == "!3":
				self.change_directive()
			elif user_input == "compress" or user_input == "!4":
				self.compress_and_clear_messages()
			elif user_input == "model" or user_input == "!5":
				self.change_model()
			elif user_input == "clear" or user_input == "!6":
				self.messages = []
			else:
				self.add_message({"role": "user", "content": user_input, "compressed": False})

				if num_tokens_from_messages(self.messages) > int(self.model["max_context"]*0.7):
					self.compress_and_clear_messages()					

				content = self.generate_response(self.directive + self.messages)
				self.add_message({"role": "assistant", "content": content, "compressed": False})
				cmd = has_tag(content)

				if self.directive_number == 2 and cmd is not None: # CmdGPT
					self.run_cmd(cmd.string)
	
				print()

	def print_help(self):
		print("This chatbot has a few reserved keywords for system management.")
		print("0. exit - Exit program.")
		print("1. help - Print help message")
		print("2. msg - Print message log.")
		print("3. change - Print an enumerated list of different directive choices.")
		print("4. compress - Compress message log")
		print("5. model - Change model")
		print("6. clear - Clear the message log.")

	def change_model(self):
		user_input = None
		while not user_input and not user_input == 0:
			for i, model in enumerate(models):
				print(i, "-", model["model"])
			print("Choose model:", end="")
			try:
				user_input = int(input())
				self.model_number = user_input
				self.model = models[self.model_number]
			except Exception as e:
				print(e)
				print("Choose a number")


	def change_directive(self):
		user_input = None
		while not user_input and not user_input == 0:
			for i, directive in enumerate(directives):
				print(i, "-", directive[0]['content'])
			print("Choose directive: ", end="")
			try:
				user_input = int(input())
				self.directive_number = user_input
				self.directive = directives[self.directive_number]
				self.directive_length = num_tokens_from_messages(self.directive)
			except Exception as e:
				print(e)
				print("Choose a number")

	def run_cmd(self, cmd):
		print()
		#print("### Running shell cmd:", cmd)
		#print()
		try:
			proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			o, e = proc.communicate()
			output = o.decode("utf-8") + e.decode("utf-8")
			self.add_message({"role": "system", "content": output, "compressed": False})
			content = self.generate_response(self.directive + self.messages)
			self.add_message({"role": "assistant", "content": content, "compressed": False})
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
		for resp in generate_response(messages, model = self.model["model"], max_context_length = self.model["max_context"]):
			try:
				delta = resp["choices"][0]["delta"]["content"]
				content = content + delta
				print(delta, end="")
			except KeyError as ke:
				pass
		return content


qti = CmdGPT()
try:
	qti.run()
except KeyboardInterrupt as ki:
	print()
	exit()
