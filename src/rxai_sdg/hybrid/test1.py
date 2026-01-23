from src.rxai_sdg.hybrid import create_reasoning_completion_generator, HybridReasoningPromptCreator, \
    HybridReasoningGenerator, TOPICS_HYBRID_REASONING
import os
import requests
from datasets import load_dataset

dataset = load_dataset("ReactiveAI/beta-reasoning", "Dolci-Think-SFT-32B", split="train")
ovh_api_key = os.environ.get("OVH_API_KEY")
url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


generator = HybridReasoningGenerator(
            max_items=1000,
            model_name="gpt-oss-20b",
            api_url=url,
            api_key=ovh_api_key,
        )
prompt_creator = HybridReasoningPromptCreator(topics=TOPICS_HYBRID_REASONING)

# Single mode
generator.generate_single(
    prompt_creator=prompt_creator,
    num_interactions=5,
    conversations=3,
    stream=False,
    additional_config={}
)
ds = generator.items
print(ds["interactions"][0])
# # All at once mode
# generator.generate_all_at_once(
#     prompt_creator=prompt_creator,
#     num_interactions=5,
#     conversations=20
# )

#
# test_dataset = dataset.select(range(10))
# generator = create_reasoning_completion_generator(api_key=ovh_api_key, api_url=url, model_name="gpt-oss-20b")
# generator.complete_all_at_once(dataset=test_dataset, additional_config={}, num_tries=3)
