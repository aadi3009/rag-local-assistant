import json
import requests
import subprocess
import time
from jinja2 import Template
from typing import List, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass

LLAMA3_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

DEFAULT_PERSONALITY_PREPROMPT = (
    {
        "role": "system",
        "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
    },
)

@dataclass
class LlamaServerConfig:
    llama_cpp_repo_path: str
    model_path: str
    context_length: int = 512
    port: int = 8080
    use_gpu: bool = True
    enable_split_mode: bool = False
    enable_flash_attn: bool = False

    @classmethod
    def from_yaml(cls, path: str, key_to_config: List[str] = ["LlamaServer"]):
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        config = data
        for nested_key in key_to_config:
            config = config.get(nested_key, {})
        if not config:
            return None
        return cls(**config)

class LlamaServer:
    def __init__(self, config: LlamaServerConfig):
        self.config = config
        self.process = None

    def start(self):
        command = [
            str(Path(self.config.llama_cpp_repo_path) / "llama-server"),
            "--model", self.config.model_path,
            "--ctx-size", str(self.config.context_length),
            "--port", str(self.config.port)
        ]
        if self.config.use_gpu:
            command += ["--n-gpu-layers", "1000"]
        if self.config.enable_split_mode:
            command += ["--split-mode", "row"]
        if self.config.enable_flash_attn:
            command += ["--flash-attn"]

        print(f"Starting LlamaServer with command: {' '.join(command)}")
        self.process = subprocess.Popen(command)

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def is_running(self):
        health_check_url = f"http://localhost:{self.config.port}/health"
        try:
            response = requests.get(health_check_url)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

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
    # Load LlamaServer configuration
    config_path = r"C:/Users/jdorr/Desktop/newglados/GlaDOS/glados_config.yml"  # Replace with your actual config path
    llama_config = LlamaServerConfig.from_yaml(config_path)
    
    if not llama_config:
        print("Failed to load LlamaServer configuration.")
        return

    # Start LlamaServer
    llama_server = LlamaServer(llama_config)
    llama_server.start()

    # Wait for the server to start
    max_wait_time = 60  # seconds
    start_time = time.time()
    while not llama_server.is_running():
        if time.time() - start_time > max_wait_time:
            print("LlamaServer failed to start within the allocated time.")
            llama_server.stop()
            return
        time.sleep(1)

    print("LlamaServer started successfully.")

    # Initialize and start the console assistant
    completion_url = f"http://localhost:{llama_config.port}/completion"
    assistant = ConsoleAssistant(completion_url)
    
    try:
        assistant.start_conversation()
    finally:
        llama_server.stop()
        print("LlamaServer stopped.")

if __name__ == "__main__":
    main()