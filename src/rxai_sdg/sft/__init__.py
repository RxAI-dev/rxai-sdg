import random
from openai import OpenAI
from datasets import Dataset, load_dataset
from datetime import datetime
from typing import Union
from ..base import BaseDatasetGenerator
from .examples import ALL_EXAMPLES_STORIES
from .prompts import ALL_PROMPTS_STORIES, PromptFactory
from .utils import common_male_names, common_female_names

class InteractionSftPromptCreator:
    def __init__(
            self,
            prompts: Union[list[PromptFactory], tuple[PromptFactory, PromptFactory, PromptFactory]] = ALL_PROMPTS_STORIES,
            examples: Union[list[str], tuple] = ALL_EXAMPLES_STORIES,
            use_first_examples_with_priority: bool = False,
    ):
        system_prompt, prompt_start, prompt_end = prompts
        self.system_prompt = system_prompt
        self.prompt_start = prompt_start
        self.prompt_end = prompt_end
        self.examples = examples
        self.use_first_examples_with_priority = use_first_examples_with_priority

    def get_examples(self, iteration: int) -> list[str]:
        if self.use_first_examples_with_priority and iteration % 3 == 0:
            return self.examples[0]
        else:
            exs = self.examples[1:] if self.use_first_examples_with_priority else self.examples
            return random.choice(exs)

    def get_system_prompt(self, num_examples: int) -> str:
        return self.system_prompt(num_examples)

    def __call__(self, iteration: int, num_examples: int = 20, include_no_think: bool = True) -> str:
        return self.prompt_start(num_examples) + self.get_examples(iteration) + self.prompt_end(num_examples, include_no_think=include_no_think)

class InteractionSftSyntheticDatasetGenerator(BaseDatasetGenerator):
    def _init_items(self):
        return { 'query': [], 'answer': [] }

    def process_items(self, items_str: str, stream: bool = False) -> int:
        try:
            items_list = self._get_items_list(items_str, stream=stream)
            items_len = len(items_list)
            for i in range(items_len):
                query, answer = items_list[i]
                self.items['query'].append(query)
                self.items['answer'].append(answer)
            return items_len
        except:
            self.failed_count += 1
            print(f'Cannot process generated list! Failed {self.failed_count} times')
            return 0

    def __call__(self, prompt_creator: InteractionSftPromptCreator, iterations: int, num_examples: int = 20, include_no_think: bool = True, stream: bool = False, temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 50, max_tokens: int = 12000, timeout: int = 120, restart: bool = False):
        if restart:
            self.items = self._init_items()

        for iteration in range(iterations):
            # Create example prompt
            prompt = prompt_creator(iteration, num_examples, include_no_think=include_no_think)
            # Get system prompt
            system_prompt = prompt_creator.get_system_prompt(num_examples)
            # Call API to generate items
            txt = self.generate_items(
                prompt, stream=stream, temperature=temperature, top_p=top_p,
                top_k=top_k, max_tokens=max_tokens, timeout=timeout, system_prompt=system_prompt,
            )
            new_items_len = self.process_items(txt, stream=stream)
            total_items = len(self.items['query'])
            if stream:
                print('\n')
            print(f'{iteration + 1}/{iterations}: Added {new_items_len} new items, total items {total_items}')
            if total_items > self.max_items:
                print('Max items limit reached, breaking.')
                break

class InteractionSftGeneratorPostprocessor:
    def __init__(self, generator: InteractionSftSyntheticDatasetGenerator, dataset_id: str, config_name: str = None, split: str = 'train',
                 token: str = None):
        self.generator = generator
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.token = token

    def get_subset(self, split_idx: int) -> InteractionSftSyntheticDatasetGenerator:
        queries_a, queries_b = self.generator.items['query'][:split_idx], self.generator.items['query'][split_idx:]
        answers_a, answers_b = self.generator.items['answer'][:split_idx], self.generator.items['answer'][split_idx:]


        self.generator.items = {
            'query': queries_a,
            'answer': answers_a,
        }

        generator_b = InteractionSftSyntheticDatasetGenerator(max_items=self.generator.max_items)

        generator_b.items = {
            'query': queries_b,
            'answer': answers_b,
        }

        return generator_b

    def get_subset_postprocessor(self, split_idx: int) -> "InteractionSftGeneratorPostprocessor":
        return self.__class__(self.get_subset(split_idx), self.dataset_id, self.config_name, self.split, self.token)

    def append_from_existing_dataset(self):
        dataset = load_dataset(self.dataset_id, self.config_name, split=self.split, token=self.token)

        self.generator.items = {
            'query': dataset['query'] + self.generator.items['query'],
            'answer': dataset['answer'] + self.generator.items['answer'],
        }

    def push_to_hf_hub(self):
        ds = self.generator.get_dataset()
        if self.config_name is not None:
            ds.push_to_hub(repo_id=self.dataset_id, config_name=self.config_name, split=self.split, token=self.token)
        else:
            ds.push_to_hub(repo_id=self.dataset_id, split=self.split, token=self.token)

    def replace_common_names(self, new_names: tuple[list[str], list[str]], existing_names: tuple[list[str], list[str]] = (common_female_names, common_male_names), skip_ratio: float = 0.0):
        ds = self.generator.get_dataset()
        processor = InteractionSftGeneratorPostprocessor.get_name_processor(*existing_names, *new_names, skip_ratio=skip_ratio)
        new_ds = ds.map(processor)
        self.generator.items = {
            'query': new_ds['query'],
            'answer': new_ds['answer'],
        }

    def ds_with_replaced_common_names(self, new_names: tuple[list[str], list[str]], existing_names: tuple[list[str], list[str]] = (common_female_names, common_male_names), skip_ratio: float = 0.0):
        ds = self.generator.get_dataset()
        processor = InteractionSftGeneratorPostprocessor.get_name_processor(*existing_names, *new_names, skip_ratio=skip_ratio)
        return ds.map(processor)

    @staticmethod
    def get_name_processor(
            female_names: list[str],
            male_names: list[str],
            new_female_names: list[str],
            new_male_names: list[str],
            skip_ratio: float = 0.0
    ):
        def process_item(item: dict) -> dict:
            query = item['query']
            answer = item['answer']
            for name in female_names:
                if name in query and name in answer:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_female_names)
                        query = query.replace(name, new_name)
                        answer = answer.replace(name, new_name)
                elif name in query:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_female_names)
                        query = query.replace(name, new_name)
                elif name in answer:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_female_names)
                        answer = answer.replace(name, new_name)
            for name in male_names:
                if name in query and name in answer:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_male_names)
                        query = query.replace(name, new_name)
                        answer = answer.replace(name, new_name)
                elif name in query:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_male_names)
                        query = query.replace(name, new_name)
                elif name in answer:
                    if random.random() > skip_ratio:
                        new_name = random.choice(new_male_names)
                        answer = answer.replace(name, new_name)
            return {'query': query, 'answer': answer}

        return process_item

class InteractionSftNameGenerator:
    syllables = [
        "Amar", "Bry", "Cass", "Dra", "Ely",
        "Fen", "Gali", "Hydr", "Iris", "Jor",
        "Kael", "Liss", "Myst", "Ner", "Ophi",
        "Per", "Quel", "Ros", "Syl", "Thal",
        "Ul", "Vael", "Xan", "Yas", "Zin",
        "Ada", "Rea", "Ye", "Men", "Jun",
        "Der", "Van", "Vir", "Mad", "Bad",
        "Mag", "Ren"
    ]

    endings = {
        "female": ["a", "elle", "in", "ara", "essa", "ila", "la", "le", "il", "ne", "ana", "na", "da"],
        "male": ["on", "ard", "thor", "rin", "ric", "am", "em", "ik", "om", "um", "ym", "od", "ad"]
    }

    @staticmethod
    def generate(n=100, gender="mixed"):
        names = []
        for _ in range(n):
            n_parts = random.randint(1, 2)
            parts = [random.choice(InteractionSftNameGenerator.syllables) for _ in range(n_parts)]
            if gender == "female":
                name = f"{''.join(parts)}{random.choice(InteractionSftNameGenerator.endings['female'])}"
            elif gender == "male":
                name = f"{''.join(parts)}{random.choice(InteractionSftNameGenerator.endings['male'])}"
            else:
                name = f"{''.join(parts)}{random.choice(InteractionSftNameGenerator.endings['female' if random.random() <0.5 else 'male'])}"
            names.append(name.title().capitalize())
        return names