import tiktoken
import openai
import pprint
import subprocess
import os
from bs4 import BeautifulSoup


try:
	openai.api_key = os.environ["OPENAI_API_KEY"]
except:
	print("You need to set the OPENAI_API_KEY environement variable to run this script.")
	exit()

MAX_CONTEXT_LENGTH = 4096

directives = [
	[{"role": "system", "content": "You are a personal assistant that answers questions as best as you can."}],
	[{"role": "system", "content": "You are a python code generator that translates natural language input from the user into python code. When the user says something, you respond in python code and nothing else."}],
	[{
		"role": "system",
		"content": """You are a natural language to unix terminal computer interface program, that takes natural language description of tasks that can be performed on the unix terminal,
					and emits a unix command on the form '<cmd>unix command</cmd>' to be ran in the terminal, you will get a response from the system of the output of the command.
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
		"content": "assistant: The content of the current working directory is 'file1.txt'."
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
	}],
]


def generate_response(messages, n=1, stream=True):
	messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=messages,
		temperature=1,
		n=n,
		max_tokens=MAX_CONTEXT_LENGTH-num_tokens_from_messages(messages),
		stream = stream
	)
	
	return response


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
	messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
	"""Returns the number of tokens used by a list of messages."""
	try:
		encoding = tiktoken.encoding_for_model(model)
	except KeyError:
		encoding = tiktoken.get_encoding("cl100k_base")
	if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
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


def has_cmd(content):
	soup = BeautifulSoup(content, "lxml")
	return soup.find('cmd')


def summarize_chat(text):
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
	response = generate_response(messages, stream=False)
	return response['choices'][0]['message']['content']


def compress_text(text):
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
	response = generate_response(messages, stream=False)
	return response['choices'][0]['message']['content']


def join_messages(messages):
	res = ""
	for i, msg in enumerate(messages):
		m = f"<msg{i}>" + msg['role'] + ": " + msg['content'] + f"</msg{i}>\n"
		res += m
	return res


class ChatGPT:

	def __init__(self):
		self.directive_number = 0
		self.directive = directives[self.directive_number]
		self.messages = []
		self.directive_length = num_tokens_from_messages(self.directive)

	def add_message(self, message):
		tokens_in_message = num_tokens_from_messages([message])
		if tokens_in_message > MAX_CONTEXT_LENGTH:
			print(message)
			print("THE MESSAGE IS TOO BIG ({} TOKENS) TO FIT IN THE CHAT, SKIPPING MESSAGE\n".format(tokens_in_message))
		elif tokens_in_message > int(MAX_CONTEXT_LENGTH*0.8):
			print("THE MESSAGE IS BIG, TRYING TO COMPRESS IT FIRST")
			compressed = compress_text(message["content"])
			message = {"role": message['role'], "content": compressed['content']}
		self.messages.append(message)

	def run(self):
		print("This chatbot has a few reserved keywords for system management.")
		print("0. exit - Exit program.")
		print("1. clear - Clear the message log.")
		print("2. msg - Print message log.")
		print("3. change - Print an enumerated list of different directive choices.")

		while True:
			print()
			print("----------------------------------------------------------------")
			print("#{} user_input ({}/{})>> ".format(self.directive_number, num_tokens_from_messages(self.messages), MAX_CONTEXT_LENGTH - self.directive_length), end="")
			user_input = input()
			print()

			if user_input == "":
				continue
			if user_input == "exit" or user_input == "!0":
				exit()
			elif user_input == "clear" or user_input == "!1":
				self.messages = []
			elif user_input == "msg" or user_input == "!2":
				for msg in self.messages:
				    pprint.pprint(msg)
			elif user_input == "change" or user_input == "!3":
				self.change_directive()
			else:
				self.add_message({"role": "user", "content": user_input, "compressed": False})

				if num_tokens_from_messages(self.messages) > int(MAX_CONTEXT_LENGTH*0.7):
					self.compress_and_clear_messages()					

				content = self.generate_response(self.directive + self.messages)
				cmd = has_cmd(content)

				if self.directive_number == 2 and cmd is not None: # CmdGPT
					self.run_cmd(cmd.string)
				else:
					self.add_message({"role": "assistant", "content": content, "compressed": False})
					print()

	def change_directive(self):
		user_input = None
		while not user_input:
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
		print("### Running shell cmd:", cmd)
		print()
		try:
			proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			o, e = proc.communicate()
			output = o.decode("utf-8") + e.decode("utf-8")
			if output == "": return
			self.add_message({"role": "system", "content": output, "compressed": False})
			content = self.generate_response(self.directive + self.messages)
			self.add_message({"role": "assistant", "content": content, "compressed": False})
		except Exception as e:
			print(e)

	def compress_and_clear_messages(self):
		to_compress = []
		while num_tokens_from_messages(self.messages) > int(MAX_CONTEXT_LENGTH*0.3):
			if not self.messages[0]['compressed']:
				to_compress.append(self.messages[0])
			self.messages.pop(0)

		joined_text = join_messages(to_compress)
		summary = summarize_chat(joined_text)
		self.messages.insert(0, {
				"role": "system",
				"content": "This is a summary of the previous conversation:\n" + summary,
				"compressed": True
			})	

	def generate_response(self, messages):
		content = ""
		for resp in generate_response(messages):
			try:
				delta = resp["choices"][0]["delta"]["content"]
				content = content + delta
				if not has_cmd(content): print(delta, end="")
			except KeyError as ke:
				pass
		return content


qti = ChatGPT()
try:
	qti.run()
except KeyboardInterrupt as ki:
	print()
	exit()
