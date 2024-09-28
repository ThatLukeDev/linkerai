from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODELPATH = "models/textmodel"

import requests
def web_get(params):
    return requests.get(params["url"]).text

FUNCLIST = {
    "web_get": web_get,
}

FUNCTIONS = [
    {
        "name": "web_get",
        "description": "Retrieve information from a webpage from the wget command line tool.",
        "parameters": {
            "type": "dict",
            "required": [
                "url"
            ],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url that will be fetched from."
                }
            },
        }
    },
]

USERHEADER = "<|start_header_id|>user<|end_header_id|>"
ASSISTANTHEADER = "<|start_header_id|>assistant<|end_header_id|>"
SYSTEMHEADER = "<|start_header_id|>system<|end_header_id|>"
ENDTOKEN = "<|eot_id|>"
CALLTOKEN = "<|call|>"

SYSTEMPROMPT = """
Cutting Knowledge Date: December 2023

When you receive a tool call response, use the output to format an answer to the orginal user question.
You are a helpful assistant with tool calling capabilities.
You should not always call a tool.

If you decide to call a tool, respond in the format:
<|call|>{"name": function name, "parameters": dictionary of argument name and its value}
Do not use variables.
If you decide to call a tool, the output will be returned to you then you will be given the opportunity to write your answer.
You must use the <|call|> tag to call a tool.
Here is a list of functions in JSON format that you can invoke.
""" + json.dumps(FUNCTIONS, indent=4)

model = AutoModelForCausalLM.from_pretrained(MODELPATH, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODELPATH)

chat = SYSTEMHEADER + SYSTEMPROMPT + ENDTOKEN

def gen():
    global chat
    chat += ASSISTANTHEADER
    input_ids = tokenizer.encode(chat, return_tensors="pt")
    output = model.generate(input_ids, max_length=65535, num_return_sequences=1)
    chat = tokenizer.decode(output[0])
    response = chat.split(ASSISTANTHEADER)[-1].replace(ENDTOKEN, "")
    if CALLTOKEN in response:
        dump = json.loads(response.split(CALLTOKEN)[-1])
        chat += FUNCLIST[dump["name"]](dump["parameters"])
        return gen()
    return response

def prompt(text):
    global chat
    chat += USERHEADER + text + ENDTOKEN
    return gen()

while True:
    print(prompt(input()))