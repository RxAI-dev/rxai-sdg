from abc import ABC, abstractmethod
from datasets import Dataset
from datetime import datetime
from ollama import Client


class SimpleOllamaTester(ABC):
    def __init__(self, model_name="llama2", api_url="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.client = Client(host=api_url)
        self.responses = []

    @abstractmethod
    def _init_items(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class WorkingTester(SimpleOllamaTester):
    def _init_items(self):
        return {"test": []}

    def __call__(self, prompt="Hello, are you there?"):
        print(f"Testing Ollama with model: {self.model_name}")
        print(f"API URL: {self.api_url}")
        print(f"Prompt: {prompt}")
        print("-" * 50)

        try:
            # Simple test
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True
            )

            print("\nResponse (streaming): ")
            full_response = ""
            for chunk in response:
                text = chunk.get('response', '')
                print(text, end="", flush=True)
                full_response += text

            self.responses.append(full_response)
            print(f"\n\n✓ Test successful!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Make sure Ollama is running: ollama serve")


# Run the test
if __name__ == "__main__":
    tester = WorkingTester(model_name="llama2")
    tester("Say 'Hello World' to confirm you're working.")