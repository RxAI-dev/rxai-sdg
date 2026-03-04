from src.rxai_sdg.hybrid import DMPOGenerator, DMPOPromptCreator, DMPOCompletionGenerator
import os
import requests
from datasets import load_dataset

ds = load_dataset("ReactiveAI/coqa-retrieval", "reasoning", split="train")
ds = ds.select(range(3))  # Select a subset for testing
do_api_key = os.environ.get("DIGITAL_OCEAN_API_KEY")
do_url = "https://inference.do-ai.run/v1"

ovh_url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
ovh_api_key = os.environ.get("OVH_API_KEY")

prompt_creator = DMPOPromptCreator()

generator = DMPOCompletionGenerator(model_name="gpt-oss-120b", api_url=ovh_url, api_key=ovh_api_key)

# generator = DMPOCompletionGenerator(model_name="mistral-nemo-instruct-2407", api_url=do_url, api_key=do_api_key)
generator(dataset=ds, mode='single', target='rejected')

