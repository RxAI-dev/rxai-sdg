from src.rxai_sdg.hybrid import DMPOGenerator, DMPOPromptCreator, DMPOCompletionGenerator
import os
import requests
from datasets import load_dataset

ds = load_dataset("ReactiveAI/coqa-retrieval", "reasoning", split="train")
ds = ds.select(range(3))  # Select a subset for testing
ovh_api_key = os.environ.get("DIGITAL_OCEAN_API_KEY")
url = "https://inference.do-ai.run/v1"

prompt_creator = DMPOPromptCreator()

generator = DMPOCompletionGenerator(model_name="openai-gpt-oss-120b", api_url=url, api_key=ovh_api_key)
generator(dataset=ds, mode='single', target='rejected')

