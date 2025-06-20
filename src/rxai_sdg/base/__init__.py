from abc import abstractmethod, ABC

from openai import OpenAI
from ollama import Client
from datasets import Dataset, load_dataset
from datetime import datetime

class BaseDatasetGenerator(ABC):
    def __init__(
            self, max_items: int = None, model_name: str = "qwen/qwen3-4b-fp8",
            api_url: str = "https://api.novita.ai/v3/openai",
            api_key: str = None,
            use_ollama: bool = False,
    ):
        self.failed_count = 0
        self.items = self._init_items()
        self.max_items = max_items
        self.use_ollama = use_ollama
        self.client = self._setup_ollama(api_url) if use_ollama else self._setup_client(api_url, api_key)
        self.model_name = model_name

    def _setup_client(self, api_url: str, api_key: str):
        return OpenAI(
            base_url=api_url,
            api_key=api_key,
        )

    def _setup_ollama(self, api_url: str):
        return Client(host=api_url)

    @abstractmethod
    def _init_items(self) -> dict[str, list]:
        pass

    def _generate_openai_like(
            self, prompt: str, stream: bool = False, temperature: float = 0.7,
            top_p: float = 0.9, top_k: int = 50, max_tokens: int = 15000,
            system_prompt: str = "", timeout: int = 120
    ):
        presence_penalty = 0
        frequency_penalty = 0
        repetition_penalty = 1
        min_p = 0
        response_format = {"type": "text"}

        chat_completion_res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
            extra_body={
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p
            },
            timeout=timeout
        )

        if stream:
            t1 = datetime.timestamp(datetime.now())
            acc = ''
            for chunk in chat_completion_res:
                if datetime.timestamp(datetime.now()) - t1 > timeout:
                    break
                ch = chunk.choices[0].delta.content or ""
                print(ch, end="")
                acc += ch
            return acc

        return chat_completion_res.choices[0].message.content

    def _generate_ollama(
            self, prompt: str, stream: bool = False, temperature: float = 0.7,
            top_p: float = 0.9, top_k: int = 50,
            system_prompt: str = "", timeout: int = 120
    ):
        chat_completion_res = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=stream,
            options={
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }
        )

        if stream:
            t1 = datetime.timestamp(datetime.now())
            acc = ''
            for chunk in chat_completion_res:
                if datetime.timestamp(datetime.now()) - t1 > timeout:
                    break
                ch = chunk['message']['content'] or ""
                print(ch, end="")
                acc += ch
            return acc

        return chat_completion_res['message']['content']
    def generate_items(
            self, prompt: str, stream: bool = False, temperature: float = 0.7,
            top_p: float = 0.9, top_k: int = 50, max_tokens: int = 15000,
            system_prompt: str = "", timeout: int = 120
    ):
        try:
            if self.use_ollama:
                return self._generate_ollama(
                    prompt, stream=stream, temperature=temperature, top_p=top_p, top_k=top_k,
                    system_prompt=system_prompt, timeout=timeout
                )
            else:
                return self._generate_openai_like(
                    prompt, stream=stream, temperature=temperature, top_p=top_p, top_k=top_k,
                    max_tokens=max_tokens, system_prompt=system_prompt, timeout=timeout
                )
        except Exception as e:
            print('API Error', e)
            return '[]'

    def get_dataset(self) -> Dataset:
        return Dataset.from_dict(self.items)

    def _get_items_list(self, items_str: str, stream: bool = False) -> list[str]:
        items_str = items_str[21:] if '<think>' in items_str else items_str
        if not items_str.rstrip().endswith(']'):
            items_str = items_str.rstrip() + ']'
        if not items_str.lstrip().startswith('['):
            items_str = '[' + items_str.lstrip()
        if not stream:
            print(items_str)
        items_list = eval(items_str)
        return items_list

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass
