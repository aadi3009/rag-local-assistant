import json
import requests
from jinja2 import Template
from typing import List, Optional

from glados.llama import LlamaServer, LlamaServerConfig

LLAMA3_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

DEFAULT_PERSONALITY_PREPROMPT = (
    {
        "role": "system",
        "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
    },
)

class ConsoleAssistant:
    def __init__(
        self,
        completion_url: str,
        api_key: Optional[str] = None,
        personality_preprompt: List[dict[str, str]] = DEFAULT_PERSONALITY_PREPROMPT,
    ):
        self.completion_url = completion_url
        self.prompt_headers = {"Authorization": api_key or "Bearer your_api_key_here"}
        self._messages = list(personality_preprompt)
        self.template = Template(LLAMA3_TEMPLATE)

    def generate_response(self, user_input: str):
        self._messages.append({"role": "user", "content": user_input})

        prompt = self.template.render(
            messages=self._messages,
            bos_token="<|begin_of_text|>",
            add_generation_prompt=True,
        )

        data = {
            "stream": True,
            "prompt": prompt,
        }

        print("Assistant: ", end="", flush=True)
        with requests.post(
            self.completion_url,
            headers=self.prompt_headers,
            json=data,
            stream=True,
        ) as response:
            assistant_response = []
            for line in response.iter_lines():
                if line:
                    line = json.loads(line.decode('utf-8').removeprefix("data: "))
                    if not line["stop"]:
                        token = line["content"]
                        print(token, end="", flush=True)
                        assistant_response.append(token)
                    else:
                        break
            
        print("\n")
        self._messages.append({"role": "assistant", "content": "".join(assistant_response)})

    def start_conversation(self):
        print("Welcome! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            self.generate_response(user_input)

def main():
    completion_url = "http://localhost:8080/completion"  # Replace with your actual URL
    api_key = None  # Replace with your API key if needed
    
    assistant = ConsoleAssistant(completion_url, api_key)
    assistant.start_conversation()

if __name__ == "__main__":
    main()