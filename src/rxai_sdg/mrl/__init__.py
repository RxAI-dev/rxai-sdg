import random
from openai import OpenAI
from datasets import Dataset, load_dataset
from datetime import datetime
from typing import Union, Callable
from ..base import BaseDatasetGenerator
from .prompts import ALL_PROMPTS_REAL
from .examples import EXAMPLES_REAL_MICRO, EXAMPLES_REAL_MICRO_LONG

class MrlPromptCreator:
    def __init__(
            self,
            prompts: Union[list[Callable], tuple[Callable, Callable, Callable, Callable]] = ALL_PROMPTS_REAL,
            examples: dict[int, str] = None,
            long_examples: dict[int, str] = None,
            topics: list[str] = None
    ):
        self.example_strs = EXAMPLES_REAL_MICRO if examples is None else examples

        self.long_example_strs = EXAMPLES_REAL_MICRO_LONG if long_examples is None else long_examples

        self.examples_start = """
    ## FEW-SHOT EXAMPLES (Do not generate same or almost the same ones)
    ```python
    """

        self.examples_end = """
    ```
    """

        self.topics = topics if topics is not None else []

        system_prompt, description, critical_rules, final_instructions = prompts
        self.system_prompt = system_prompt
        self.description = description
        self.critical_rules = critical_rules
        self.final_instructions = final_instructions

    def get_follow_ups_format(self, steps: int, mode: str = 'multi'):
        if mode == 'long':
            format_follow_ups = f"""
              [
                  ("[Follow-up 1: **topic two**]", "[Answer 1: **topic two**]"),
                  ("[Follow-up 2: **topic two**]", "[Answer 2: **topic two**]"),
                  # ... {steps - 1} examples for **topic two**
                  ("[Follow-up {steps}: **topic one** (final)]", "[Answer {steps}: **topic one**  (final)]"),
              ]
        """
        elif steps == 1:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
              ]
        """
        elif steps == 2:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
                  ("[Follow-up 2]", "[Answer 2]"),
              ]
        """
        else:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
                  ("[Follow-up 2]", "[Answer 2]"),
                  # ... {steps} total
              ]
        """
        return format_follow_ups

    def get_description(self, steps: int, num_examples: int, prior_steps: int, mode: str = 'multi'):
        return self.description(self.get_follow_ups_format(steps, mode), steps, num_examples, prior_steps, mode)


    def get_critical_rules(self, steps: int, prior_steps: int, num_tokens: int, mode: str = 'multi'):
        return self.critical_rules(steps, prior_steps, num_tokens, mode)

    def get_examples(self, steps: int, mode: str = 'multi'):
        exs = self.example_strs[steps] if mode == 'multi' else self.long_example_strs[steps]
        return self.examples_start + exs + self.examples_end

    def get_topics(self, num_topics: int, mode: str = 'multi'):
        topics = random.choices(self.topics, k=num_topics)

        if num_topics != 0:
            topics_bullets = '/n'.join([f'- {topic}' for topic in topics])

            topics_str = f"""
      ## TOPICS FOR GENERATED EXAMPLES:
        - do not use same examples like FEW SHOTS, try different topics, like following ones
        - you can use one of the following topics for generated examples or similar one
        {topics_bullets}
      """ if mode == 'multi' else f"""
      ## TOPICS FOR GENERATED EXAMPLES:
      - do not use same examples like FEW SHOTS, try different **topics**, like following ones
      - you should use **TWO** of the following **topics** for generated examples or some similar **topics**
      {topics_bullets}
      """
        else:
            topics_str = ''

        return topics_str

    def get_final_instructions(self, steps: int, num_examples: int, include_no_think: bool = True, mode: str = 'multi'):
        instructions = self.final_instructions(steps, num_examples, mode=mode)

        no_think = """
        /no_think
        """

        return instructions + no_think if include_no_think else instructions

    def get_prior_steps(self, steps: int):
        if steps < 4:
            prior_steps = 1
        elif steps >= 4:
            prior_steps = 2
        elif steps <= 8:
            prior_steps = 3
        else:
            prior_steps = 4

        return prior_steps

    def get_system_prompt(self, num_examples: int):
        return self.system_prompt(num_examples)

    def __call__(self, steps: int, num_examples: int = 10, num_topics: int = 10, mode: str = 'multi',
                 include_no_think: bool = True, num_tokens: int = 256):
        prior_steps = self.get_prior_steps(steps)
        return (self.get_description(steps, num_examples, prior_steps, mode=mode) +
                self.get_critical_rules(steps, prior_steps, num_tokens, mode=mode) +
                self.get_examples(steps, mode=mode) + self.get_topics(num_topics, mode=mode) +
                self.get_final_instructions(steps, num_examples, include_no_think=include_no_think, mode=mode))


class MrlSyntheticDatasetGenerator(BaseDatasetGenerator):
    def _init_items(self):
        return {'query': [], 'answer': [], 'interactions': []}

    def filter_incorrect_long_range(self, query: str, answer: str,
                                    interactions: list[tuple[str, str]]) -> bool:
        from nltk.translate.bleu_score import sentence_bleu
        initial = f'[Q] {query.strip()} [A] {answer.strip()}'
        follow_up = [f"[Q] {query.strip()} [A] {answer.strip()}" for query, answer in interactions]
        topic_two_follow_ups, last_follow_up = follow_up[:-1], follow_up[-1]

        bleu_initial_last = (sentence_bleu([last_follow_up], initial, weights=(0.25, 0.25, 0.25, 0.25)) + sentence_bleu(
            [initial], last_follow_up, weights=(0.25, 0.25, 0.25, 0.25))) / 2

        has_one_middle_incorrect = False

        middle_init_bleus = []
        for item in topic_two_follow_ups:
            score = sentence_bleu([initial], item, weights=(0.25, 0.25, 0.25, 0.25))
            if score > bleu_initial_last:
                has_one_middle_incorrect = True
            middle_init_bleus.append(score)
        bleu_initial_follow_ups = sum(middle_init_bleus) / len(topic_two_follow_ups)

        middle_last_bleus = []
        for item in topic_two_follow_ups:
            score = sentence_bleu([last_follow_up], item, weights=(0.25, 0.25, 0.25, 0.25))
            if score > bleu_initial_last:
                has_one_middle_incorrect = True

            middle_last_bleus.append(score)

        bleu_last_follow_ups = sum(sentence_bleu([last_follow_up], item, weights=(0.25, 0.25, 0.25, 0.25)) for item in
                                   topic_two_follow_ups) / len(topic_two_follow_ups)

        if bleu_initial_last > bleu_initial_follow_ups and bleu_initial_last > bleu_last_follow_ups:
            if bleu_last_follow_ups > 0.45 and bleu_initial_follow_ups > 0.45:
                print(
                    f"Incorrect items - mixed last/initial with middle. Initial vs last: {bleu_initial_last} | Initial vs middle: {bleu_initial_follow_ups} | Last vs middle: {bleu_last_follow_ups}")
                return False
            elif bleu_last_follow_ups > 0.45:
                print(
                    f"Incorrect items - mixed last with middle. Initial vs last: {bleu_initial_last} | Initial vs middle: {bleu_initial_follow_ups} | Last vs middle: {bleu_last_follow_ups}")
                return False
            elif bleu_initial_follow_ups > 0.45:
                print(
                    f"Incorrect items - mixed initial with middle. Initial vs last: {bleu_initial_last} | Initial vs middle: {bleu_initial_follow_ups} | Last vs middle: {bleu_last_follow_ups}")
                return False
            elif has_one_middle_incorrect:
                print(
                    f"Incorrect items - some middle item connected to first topic. Initial vs last: {bleu_initial_last} | Initial vs middle: {middle_init_bleus} | Last vs middle: {middle_last_bleus}")
                return False
            return True
        else:
            print(
                f"Incorrect items - wrong topics. Initial vs last: {bleu_initial_last} | Initial vs middle: {bleu_initial_follow_ups} | Last vs middle: {bleu_last_follow_ups}")
            return False

    def filter_incorrect(self, steps: int, query: str, answer: str,
                         interactions: list[tuple[str, str]], mode: str = 'multi') -> bool:
        if mode == 'multi':
            return query is not None and answer is not None and interactions is not None and len(interactions) == steps
        else:
            return query is not None and answer is not None and interactions is not None and len(
                interactions) == steps and self.filter_incorrect_long_range(query, answer, interactions)

    def process_interactions(self, interactions: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{'query': query.strip(), 'answer': answer.strip()} for query, answer in interactions]

    def process_items(self, items_str: str, steps: int, stream: bool = False, mode: str = 'multi') -> int:
        try:
            items_list = self._get_items_list(items_str, stream=stream)
            total_items_len = len(self.items['query'])
            for (query, answer), interactions in items_list:
                if self.filter_incorrect(steps, query, answer, interactions, mode=mode):
                    self.items['query'].append(query.strip())
                    self.items['answer'].append(answer.strip())
                    self.items['interactions'].append(self.process_interactions(interactions))
            return len(self.items['query']) - total_items_len
        except:
            self.failed_count += 1
            print(f'Cannot process generated list! Failed {self.failed_count} times')
            return 0

    def __call__(self, prompt_creator: MrlPromptCreator, steps: int, iterations: int, num_examples: int = 10,
                 num_topics: int = 10, include_no_think: bool = True, mode: str = 'multi', stream: bool = False,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, max_tokens: int = 15000,
                 timeout: int = 120, restart: bool = False):
        if restart:
            self.items = self._init_items()

        for iteration in range(iterations):
            # Create example prompt
            prompt = prompt_creator(steps, num_examples=num_examples, num_topics=num_topics,
                                    include_no_think=include_no_think, mode=mode)
            # Get system prompt
            system_prompt = prompt_creator.get_system_prompt(num_examples)
            # Call API to generate items
            txt = self.generate_items(
                prompt, stream=stream, temperature=temperature, top_p=top_p,
                top_k=top_k, max_tokens=max_tokens, timeout=timeout, system_prompt=system_prompt,
            )
            new_items_len = self.process_items(txt, steps, stream=stream, mode=mode)
            total_items = len(self.items['query'])
            if stream:
                print('\n')
            print(f'{iteration + 1}/{iterations}: Added {new_items_len} new items, total items {total_items}')
            if total_items > self.max_items:
                print('Max items limit reached, breaking.')
                break


class MrlGeneratorPostprocessor:
    def __init__(self, generator: MrlSyntheticDatasetGenerator, dataset_id: str, config_name: str, split: str = 'train',
                 token: str = None):
        self.generator = generator
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.token = token

    def filter_duplicates(self):
        queries = []
        answers = []
        interactions = []

        items_len = len(self.generator.items['query'])

        print(f'Original size: {items_len}')

        for i in range(items_len):
            query = self.generator.items['query'][i]
            if not query in queries:
                queries.append(query)
                answers.append(self.generator.items['answer'][i])
                interactions.append(self.generator.items['interactions'][i])

        print(f'Filtered size: {len(queries)}')

        self.generator.items = {
            'query': queries,
            'answer': answers,
            'interactions': interactions
        }

    def remove_incorrect_interactions(self, steps: int):
        queries = []
        answers = []
        interactions = []

        counts = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
        }

        items_len = len(self.generator.items['query'])

        print(f'Original size: {items_len}')

        for i in range(items_len):
            item_inters = self.generator.items['interactions'][i]
            item_inters_len = len(item_inters)
            counts[item_inters_len] += 1

            if item_inters_len == steps:
                queries.append(self.generator.items['query'][i])
                answers.append(self.generator.items['answer'][i])
                interactions.append(item_inters)

        print(f"Unique interaction len counts: {[(l, c) for l, c in counts.items() if c > 0]}")

        print(f'Filtered size: {len(queries)}')

        self.generator.items = {
            'query': queries,
            'answer': answers,
            'interactions': interactions
        }

    def get_subset(self, split_idx: int) -> MrlSyntheticDatasetGenerator:
        queries_a, queries_b = self.generator.items['query'][:split_idx], self.generator.items['query'][split_idx:]
        answers_a, answers_b = self.generator.items['answer'][:split_idx], self.generator.items['answer'][split_idx:]
        interactions_a, interactions_b = self.generator.items['interactions'][:split_idx], self.generator.items[
                                                                                               'interactions'][
                                                                                           split_idx:]

        self.generator.items = {
            'query': queries_a,
            'answer': answers_a,
            'interactions': interactions_a,
        }

        generator_b = MrlSyntheticDatasetGenerator(max_items=self.generator.max_items)

        generator_b.items = {
            'query': queries_b,
            'answer': answers_b,
            'interactions': interactions_b,
        }

        return generator_b

    def get_subset_postprocessor(self, split_idx: int) -> "MrlGeneratorPostprocessor":
        return self.__class__(self.get_subset(split_idx), self.dataset_id, self.config_name, self.split, self.token)

    def append_from_existing_dataset(self):
        dataset = load_dataset(self.dataset_id, self.config_name, split=self.split, token=self.token)

        self.generator.items = {
            'query': dataset['query'] + self.generator.items['query'],
            'answer': dataset['answer'] + self.generator.items['answer'],
            'interactions': dataset['interactions'] + self.generator.items['interactions']
        }

    def push_to_hf_hub(self):
        ds = self.generator.get_dataset()
        ds.push_to_hub(repo_id=self.dataset_id, config_name=self.config_name, split=self.split, token=self.token)
