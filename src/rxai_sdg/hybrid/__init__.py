"""
RxT-Beta Hybrid Reasoning Dataset Generators

This module provides generators for creating synthetic datasets for training
RxT-Beta with hybrid reasoning capabilities:

1. ReasoningCompletionGenerator - Add missing 'think' blocks to existing conversations
2. HybridReasoningGenerator - Create new conversations with full reasoning chains
3. DMPOGenerator - Create preference pairs for memory optimization training

All generators support both OpenAI-compatible APIs and Ollama for local testing.

RxT-Beta Interaction Template:
- [Q] query [A] answer - Fast answer mode
- [Q] query [T] thinking [A] answer - Extended thinking mode
- [U] tool_result [T] thinking [A] answer - Tool usage mode
- [I] instruction - Internal instruction
- [C] tool_call - Agentic tool call
"""

import random
import json
from typing import Union, Callable, Literal, Optional
from datasets import Dataset, load_dataset

from ..base import BaseDatasetGenerator
from .prompts import (
    TOPICS_HYBRID_REASONING,
    system_reasoning_completion_single,
    system_reasoning_completion_all,
    task_description_reasoning_completion_single,
    task_description_reasoning_completion_all,
    system_reasoning_generation_single,
    system_reasoning_generation_all,
    task_description_reasoning_generation_single,
    task_description_reasoning_generation_all,
    system_dmpo_generation_single,
    system_dmpo_generation_all,
    task_description_dmpo_single,
    task_description_dmpo_all,
    get_random_topics,
)
from .examples import (
    get_reasoning_completion_example_single,
    get_reasoning_completion_example_all,
    get_reasoning_generation_example_single,
    get_reasoning_generation_example_all,
    get_dmpo_example_single,
    get_dmpo_example_all,
)
from ..base.test import ollama_api_key


# ============================================================================
# REASONING COMPLETION GENERATORS
# For adding missing 'think' blocks to existing conversations
# ============================================================================

class ReasoningCompletionGenerator(BaseDatasetGenerator):
    """
    Generator for completing missing reasoning/thinking blocks in conversations.

    This generator takes conversations where only the final interaction has a
    'think' block and generates the missing thinking blocks for all previous
    interactions.

    Supports two modes:
    - single: Generate one think block at a time with full context
    - all_at_once: Generate all think blocks for a conversation at once

    Example usage:
        generator = ReasoningCompletionGenerator(
            max_items=100,
            model_name="gpt-4",
            api_url="https://api.openai.com/v1",
            api_key="your-key"
        )

        # Single mode
        generator.complete_single(
            dataset=hf_dataset,
            target_tokens=512,
            temperature=0.7
        )

        # All at once mode
        generator.complete_all_at_once(
            dataset=hf_dataset,
            target_tokens=512,
            temperature=0.7
        )
    """

    def _init_items(self) -> dict[str, list]:
        """Initialize storage for completed conversations."""
        return {'interactions': []}

    def _parse_think_block(self, response: str) -> str:
        """Extract and clean thinking content from API response."""
        # Remove any wrapping markers
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        return response

    def _parse_think_list(self, response: str) -> list[str]:
        """Parse a Python list of thinking blocks from API response."""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        if response.endswith('}'):
            response = response[:-1]
        response = response.strip()

        # Try to parse as Python list
        try:
            result = eval(response)
            if isinstance(result, list):
                return [str(item) for item in result]
        except:
            pass

        # Fallback: try json
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return [str(item) for item in result]
        except:
            pass

        return []

    def complete_single(
        self,
        dataset: Dataset,
        target_tokens: int = 512,
        iterations: int = None,
        stream: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        timeout: int = 180,
        additional_config: dict = None,
        include_examples: bool = True,
        num_tries: int = 3
    ):
        """
        Generate missing think blocks one at a time with full context.

        Args:
            dataset: HuggingFace dataset with 'interactions' field containing
                     list of dicts with 'query', 'think', 'answer' keys
            target_tokens: Target length for each think block
            iterations: Max number of conversations to process (None = all)
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens per generation
            timeout: Request timeout in seconds
            additional_config: Additional API configuration
            include_examples: Whether to include few-shot examples
            num_tries: Number of tries to generate model response
        """
        if iterations is None:
            iterations = len(dataset)

        system_prompt = system_reasoning_completion_single()

        for conv_idx in range(min(iterations, len(dataset))):
            conversation = dataset[conv_idx]['interactions']
            completed_interactions = []

            for step_idx, interaction in enumerate(conversation):
                query = interaction.get('query', '')
                answer = interaction.get('answer', '')
                existing_think = interaction.get('think', '')

                # If think already exists and is non-empty, keep it
                if existing_think and len(existing_think.strip()) > 10:
                    completed_interactions.append({
                        'query': query,
                        'think': existing_think,
                        'answer': answer
                    })
                    continue

                # Build memory context from prior interactions
                memory_context = completed_interactions if step_idx > 0 else None

                # Build prompt
                prompt = task_description_reasoning_completion_single(
                    query=query,
                    answer=answer,
                    memory_context=memory_context,
                    target_tokens=target_tokens
                )

                # Add few-shot example
                if include_examples:
                    example_style = "memory_reference" if memory_context else "basic"
                    example = get_reasoning_completion_example_single(example_style)
                    prompt = f"## FEW-SHOT EXAMPLE\n{example}\n\n{prompt}"

                for attempt in range(num_tries):
                    # Generate think block
                    response = self.generate_items(
                        prompt,
                        stream=stream,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        timeout=timeout,
                        additional_config=additional_config
                    )
                    if response is not None:
                        think_block = self._parse_think_block(response)

                    if len(think_block) > 0:
                        break
                    print(f"Attempt {attempt + 1} failed...\n")
                completed_interactions.append({
                    'query': query,
                    'think': think_block,
                    'answer': answer
                })

                if stream:
                    print('\n')

            # Store completed conversation
            self.items['interactions'].append(completed_interactions)
            print(f"Completed conversation {conv_idx + 1}/{iterations} ({len(completed_interactions)} interactions)")

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def complete_all_at_once(
        self,
        dataset: Dataset,
        target_tokens_per_think: int = 512,
        iterations: int = None,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 16000,
        timeout: int = 300,
        additional_config: dict = None,
        include_examples: bool = True,
        num_tries: int = 5,
        skip_last_interaction: bool = False
    ):
        """
        Generate all missing think blocks for a conversation at once.

        Args:
            dataset: HuggingFace dataset with 'interactions' field
            target_tokens_per_think: Target length for each think block
            iterations: Max conversations to process (None = all)
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens for generation
            timeout: Request timeout in seconds
            additional_config: Additional API configuration
            include_examples: Whether to include few-shot examples
            num_tries: Number of tries to generate correct parsable response
            skip_last_interaction: Optionally skip generation of the think block in the last interaction
        """
        if iterations is None:
            iterations = len(dataset)

        system_prompt = system_reasoning_completion_all()

        for conv_idx in range(min(iterations, len(dataset))):
            conversation = dataset[conv_idx]['interactions']

            # Build interaction list (without think blocks)
            if skip_last_interaction:
                interactions = [
                    {'query': inter.get('query', ''), 'answer': inter.get('answer', '')}
                    for inter in conversation[:-1]
                ]
            else:
                interactions = [
                    {'query': inter.get('query', ''), 'answer': inter.get('answer', '')}
                    for inter in conversation
                ]

            # Build prompt
            prompt = task_description_reasoning_completion_all(
                interactions=interactions,
                target_tokens_per_think=target_tokens_per_think
            )

            # Add few-shot example
            if include_examples:
                example = get_reasoning_completion_example_all()
                prompt = f"## FEW-SHOT EXAMPLE\n{example}\n\n{prompt}"

            for attempt in range(num_tries):
                # Generate all think blocks
                response = self.generate_items(
                    prompt,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    timeout=timeout,
                    additional_config=additional_config
                )
                think_blocks = self._parse_think_list(response)
                if len(think_blocks) == len(interactions):
                    break
                print(f"Attempt {attempt + 1}/{num_tries} failed, retrying...")


            if stream:
                print('\n')

            # Validate and store
            if len(think_blocks) == len(interactions):
                completed = [
                    {
                        'query': interactions[i]['query'],
                        'think': think_blocks[i],
                        'answer': interactions[i]['answer']
                    }
                    for i in range(len(interactions))
                ]
                completed.append(conversation[-1])
                self.items['interactions'].append(completed)
                print(f"Completed conversation {conv_idx + 1}/{iterations} ({len(completed)} interactions)")
            else:
                print(f"Warning: Got {len(think_blocks)} think blocks for {len(interactions)} interactions, skipping")
                self.failed_count += 1

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def __call__(
        self,
        dataset: Dataset,
        mode: Literal['single', 'all_at_once'] = 'single',
        **kwargs
    ):
        """
        Run reasoning completion in specified mode.

        Args:
            dataset: HuggingFace dataset with 'interactions' field
            mode: 'single' or 'all_at_once'
            **kwargs: Additional arguments passed to the specific method
        """
        if mode == 'single':
            self.complete_single(dataset, **kwargs)
        elif mode == 'all_at_once':
            self.complete_all_at_once(dataset, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'all_at_once'")


# ============================================================================
# HYBRID REASONING DATASET GENERATORS
# For creating new conversations with hybrid reasoning from scratch
# ============================================================================

class HybridReasoningPromptCreator:
    """
    Creates prompts for hybrid reasoning dataset generation.

    Supports both single interaction generation and full conversation generation.
    """

    def __init__(
        self,
        topics: list[str] = None,
        include_examples: bool = True
    ):
        self.topics = topics if topics is not None else TOPICS_HYBRID_REASONING
        self.include_examples = include_examples

    def get_single_prompt(
        self,
        topic: str,
        step_num: int,
        total_steps: int,
        prior_interactions: list[dict] = None,
        target_tokens: int = 1024,
        require_extended_thinking: bool = True,
        language: str = "English"
    ) -> tuple[str, str]:
        """
        Get prompt for single interaction generation.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = system_reasoning_generation_single()

        user = task_description_reasoning_generation_single(
            topic=topic,
            step_num=step_num,
            total_steps=total_steps,
            prior_interactions=prior_interactions,
            target_tokens=target_tokens,
            require_extended_thinking=require_extended_thinking,
            language=language
        )

        if self.include_examples:
            example = get_reasoning_generation_example_single(step_num)
            user = f"## FEW-SHOT EXAMPLE\n{example}\n\n{user}"

        return system, user

    def get_all_at_once_prompt(
        self,
        topic: str,
        num_interactions: int,
        target_tokens_per_interaction: int = 1024,
        thinking_ratio: float = 0.7,
        language: str = "English"
    ) -> tuple[str, str]:
        """
        Get prompt for full conversation generation.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = system_reasoning_generation_all()

        user = task_description_reasoning_generation_all(
            topic=topic,
            num_interactions=num_interactions,
            target_tokens_per_interaction=target_tokens_per_interaction,
            thinking_ratio=thinking_ratio,
            language=language
        )

        if self.include_examples:
            example = get_reasoning_generation_example_all(num_interactions)
            user = f"## FEW-SHOT EXAMPLE\n{example}\n\n{user}"

        return system, user

    def get_random_topic(self) -> str:
        """Get a random topic for generation."""
        return random.choice(self.topics)


class HybridReasoningGenerator(BaseDatasetGenerator):
    """
    Generator for creating hybrid reasoning conversations from scratch.

    Creates multi-turn conversations with:
    - Query: User's question or task
    - Think: Model's reasoning process (for extended thinking mode)
    - Answer: Model's final response

    Supports two modes:
    - single: Generate one interaction at a time, building conversation progressively
    - all_at_once: Generate entire conversation at once

    Example usage:
        generator = HybridReasoningGenerator(
            max_items=100,
            model_name="gpt-4",
            api_url="https://api.openai.com/v1",
            api_key="your-key"
        )

        prompt_creator = HybridReasoningPromptCreator(topics=my_topics)

        # Single mode
        generator.generate_single(
            prompt_creator=prompt_creator,
            num_interactions=5,
            conversations=20
        )

        # All at once mode
        generator.generate_all_at_once(
            prompt_creator=prompt_creator,
            num_interactions=5,
            conversations=20
        )
    """

    def _init_items(self) -> dict[str, list]:
        """Initialize storage for generated conversations."""
        return {'interactions': [], 'topics': []}

    def _parse_interaction(self, response: str) -> Optional[dict]:
        """Parse a single interaction dictionary from API response."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        # if response.endswith('}'):
        #     response = response[:-1]
        response = response.strip()

        # Try Python eval
        try:
            result = eval(response)
            if isinstance(result, dict) and 'query' in result and 'answer' in result:
                return {
                    'query': str(result.get('query', '')),
                    'think': str(result.get('think', '')),
                    'answer': str(result.get('answer', ''))
                }
        except Exception as e:
            print(e)

        # Try JSON
        try:
            result = json.loads(response)
            if isinstance(result, dict) and 'query' in result and 'answer' in result:
                return {
                    'query': str(result.get('query', '')),
                    'think': str(result.get('think', '')),
                    'answer': str(result.get('answer', ''))
                }
        except:
            pass

        return None

    def _parse_conversation(self, response: str) -> list[dict]:
        """Parse a full conversation list from API response."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = response.strip()

        # Try to fix truncated responses
        if not response.endswith(']'):
            # Try to find last complete dict
            last_complete = response.rfind('}')
            if last_complete > 0:
                response = response[:last_complete + 1] + ']'

        # Try Python eval
        try:
            result = eval(response)
            if isinstance(result, list):
                return [
                    {
                        'query': str(item.get('query', '')),
                        'think': str(item.get('think', '')),
                        'answer': str(item.get('answer', ''))
                    }
                    for item in result
                    if isinstance(item, dict) and 'query' in item and 'answer' in item
                ]
        except:
            pass

        # Try JSON
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return [
                    {
                        'query': str(item.get('query', '')),
                        'think': str(item.get('think', '')),
                        'answer': str(item.get('answer', ''))
                    }
                    for item in result
                    if isinstance(item, dict) and 'query' in item and 'answer' in item
                ]
        except:
            pass

        return []

    def generate_single(
        self,
        prompt_creator: HybridReasoningPromptCreator,
        num_interactions: int,
        conversations: int,
        target_tokens: int = 1024,
        thinking_ratio: float = 0.7,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        timeout: int = 180,
        additional_config: dict = None,
        restart: bool = False,
        language: str = "English"
    ):
        """
        Generate conversations one interaction at a time.

        Args:
            prompt_creator: HybridReasoningPromptCreator instance
            num_interactions: Number of interactions per conversation
            conversations: Number of conversations to generate
            target_tokens: Target tokens per interaction
            thinking_ratio: Ratio of interactions that use extended thinking
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens per generation
            timeout: Request timeout
            additional_config: Additional API config
            restart: Whether to clear existing items
            language: Language of generated response
        """
        if restart:
            self.items = self._init_items()

        for conv_idx in range(conversations):
            topic = prompt_creator.get_random_topic()
            current_conversation = []

            for step in range(1, num_interactions + 1):
                # Decide if this step uses extended thinking
                use_thinking = random.random() < thinking_ratio

                system, user = prompt_creator.get_single_prompt(
                    topic=topic,
                    step_num=step,
                    total_steps=num_interactions,
                    prior_interactions=current_conversation if step > 1 else None,
                    target_tokens=target_tokens,
                    require_extended_thinking=use_thinking,
                    language=language
                )
                max_tries = 3
                for attempt in range(max_tries):
                    response = self.generate_items(
                        user,
                        stream=stream,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        system_prompt=system,
                        timeout=timeout,
                        additional_config=additional_config
                    )

                    if stream:
                        print('\n')

                    interaction = self._parse_interaction(response)

                    if interaction is not None:
                        break

                if interaction:
                    current_conversation.append(interaction)
                else:
                    print(f"Failed to parse interaction {step} for conversation {conv_idx + 1}")
                    self.failed_count += 1
                    break

            # Only store complete conversations
            if len(current_conversation) == num_interactions:
                self.items['interactions'].append(current_conversation)
                self.items['topics'].append(topic)
                print(f"Generated conversation {conv_idx + 1}/{conversations} on '{topic[:50]}...'")
            else:
                print(f"Incomplete conversation {conv_idx + 1}, discarding")

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def generate_all_at_once(
        self,
        prompt_creator: HybridReasoningPromptCreator,
        num_interactions: int,
        conversations: int,
        target_tokens_per_interaction: int = 1024,
        thinking_ratio: float = 0.7,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 32000,
        timeout: int = 300,
        additional_config: dict = None,
        restart: bool = False,
        max_tries: int = 3,
        language: str = "English"
    ):
        """
        Generate entire conversations at once.

        Args:
            prompt_creator: HybridReasoningPromptCreator instance
            num_interactions: Number of interactions per conversation
            conversations: Number of conversations to generate
            target_tokens_per_interaction: Target tokens per interaction
            thinking_ratio: Ratio of interactions with extended thinking
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens for entire generation
            timeout: Request timeout
            additional_config: Additional API config
            restart: Whether to clear existing items
            max_tries: Number of tries to generate response - added in case model generate
            wrong response, which is incompatible with eval() function
            language: Which language to generate in.
        """
        if restart:
            self.items = self._init_items()

        for conv_idx in range(conversations):
            topic = prompt_creator.get_random_topic()

            system, user = prompt_creator.get_all_at_once_prompt(
                topic=topic,
                num_interactions=num_interactions,
                target_tokens_per_interaction=target_tokens_per_interaction,
                thinking_ratio=thinking_ratio,
                language=language
            )

            for attempt in range(max_tries):
                response = self.generate_items(
                    user,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    system_prompt=system,
                    timeout=timeout,
                    additional_config=additional_config
                )

                if stream:
                    print('\n')

                conversation = self._parse_conversation(response)
                if conversation is not None and len(conversation) > 0:
                    break

            if len(conversation) >= num_interactions * 0.8:  # Allow some tolerance
                # Pad or truncate to exact length
                while len(conversation) < num_interactions:
                    conversation.append({
                        'query': '',
                        'think': '',
                        'answer': ''
                    })
                conversation = conversation[:num_interactions]

                self.items['interactions'].append(conversation)
                self.items['topics'].append(topic)
                print(f"Generated conversation {conv_idx + 1}/{conversations} on '{topic[:50]}...' ({len(conversation)} interactions)")
            else:
                print(f"Insufficient interactions ({len(conversation)}/{num_interactions}) for conversation {conv_idx + 1}")
                self.failed_count += 1

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def __call__(
        self,
        prompt_creator: HybridReasoningPromptCreator,
        num_interactions: int,
        conversations: int,
        mode: Literal['single', 'all_at_once'] = 'single',
        **kwargs
    ):
        """
        Generate hybrid reasoning conversations.

        Args:
            prompt_creator: HybridReasoningPromptCreator instance
            num_interactions: Interactions per conversation
            conversations: Number of conversations
            mode: 'single' or 'all_at_once'
            **kwargs: Additional arguments for the specific method
        """
        if mode == 'single':
            self.generate_single(
                prompt_creator=prompt_creator,
                num_interactions=num_interactions,
                conversations=conversations,
                **kwargs
            )
        elif mode == 'all_at_once':
            self.generate_all_at_once(
                prompt_creator=prompt_creator,
                num_interactions=num_interactions,
                conversations=conversations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'all_at_once'")


# ============================================================================
# DMPO (Direct Memory and Preference Optimization) GENERATORS
# For creating preference pairs with accepted/rejected responses
# ============================================================================

class DMPOPromptCreator:
    """
    Creates prompts for DMPO dataset generation.

    Generates preference pairs where:
    - Accepted: High-quality response with good memory usage
    - Rejected: Lower-quality response with memory errors or weak reasoning
    """

    def __init__(
        self,
        topics: list[str] = None,
        include_examples: bool = True
    ):
        self.topics = topics if topics is not None else TOPICS_HYBRID_REASONING
        self.include_examples = include_examples

    def get_single_prompt(
        self,
        topic: str,
        step_num: int,
        total_steps: int,
        prior_interactions: list[dict] = None,
        target_tokens: int = 1024
    ) -> tuple[str, str]:
        """
        Get prompt for single DMPO pair generation.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = system_dmpo_generation_single()

        user = task_description_dmpo_single(
            topic=topic,
            step_num=step_num,
            total_steps=total_steps,
            prior_interactions=prior_interactions,
            target_tokens=target_tokens
        )

        if self.include_examples:
            example = get_dmpo_example_single(step_num)
            user = f"## FEW-SHOT EXAMPLE\n{example}\n\n{user}"

        return system, user

    def get_all_at_once_prompt(
        self,
        topic: str,
        num_interactions: int,
        target_tokens_per_interaction: int = 1024
    ) -> tuple[str, str]:
        """
        Get prompt for full DMPO conversation generation.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = system_dmpo_generation_all()

        user = task_description_dmpo_all(
            topic=topic,
            num_interactions=num_interactions,
            target_tokens_per_interaction=target_tokens_per_interaction
        )

        if self.include_examples:
            example = get_dmpo_example_all(num_interactions)
            user = f"## FEW-SHOT EXAMPLE\n{example}\n\n{user}"

        return system, user

    def get_random_topic(self) -> str:
        """Get a random topic for generation."""
        return random.choice(self.topics)


class DMPOGenerator(BaseDatasetGenerator):
    """
    Generator for Direct Memory and Preference Optimization datasets.

    Creates preference pairs with:
    - Query: User's question
    - Accepted: High-quality response with good memory usage
        - think: Clear, accurate reasoning
        - answer: Helpful, correct response
    - Rejected: Lower-quality response
        - think: Flawed reasoning or memory errors
        - answer: Response with issues

    This data trains the model to prefer responses that properly utilize
    memory and produce high-quality outputs.

    Example usage:
        generator = DMPOGenerator(
            max_items=100,
            model_name="gpt-4",
            api_url="https://api.openai.com/v1",
            api_key="your-key"
        )

        prompt_creator = DMPOPromptCreator(topics=my_topics)

        # Single mode
        generator.generate_single(
            prompt_creator=prompt_creator,
            num_interactions=5,
            conversations=20
        )
    """

    def _init_items(self) -> dict[str, list]:
        """Initialize storage for DMPO pairs."""
        return {'interactions': [], 'topics': []}

    def _parse_dmpo_pair(self, response: str) -> Optional[dict]:
        """Parse a single DMPO pair from API response."""
        response = response.strip()
        # Remove markdown code blocks
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = f'''{response}'''.strip()

        # Try Python eval
        try:
            result = eval( f'''{response}''')
            if isinstance(result, dict) and 'query' in result and 'accepted' in result and 'rejected' in result:
                return {
                    'query': str(result.get('query', '')),
                    'accepted': {
                        'think': str(result.get('accepted', {}).get('think', '')),
                        'answer': str(result.get('accepted', {}).get('answer', ''))
                    },
                    'rejected': {
                        'think': str(result.get('rejected', {}).get('think', '')),
                        'answer': str(result.get('rejected', {}).get('answer', ''))
                    }
                }
        except:
            pass

        # Try JSON
        try:
            result = json.loads(response)
            if isinstance(result, dict) and 'query' in result and 'accepted' in result and 'rejected' in result:
                return {
                    'query': str(result.get('query', '')),
                    'accepted': {
                        'think': str(result.get('accepted', {}).get('think', '')),
                        'answer': str(result.get('accepted', {}).get('answer', ''))
                    },
                    'rejected': {
                        'think': str(result.get('rejected', {}).get('think', '')),
                        'answer': str(result.get('rejected', {}).get('answer', ''))
                    }
                }
        except:
            pass

        return None

    def _parse_dmpo_conversation(self, response: str) -> list[dict]:
        """Parse a full DMPO conversation from API response."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith('```python'):
            response = response[9:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = response.strip()

        # Try to fix truncated responses
        if not response.endswith(']'):
            last_complete = response.rfind('}')
            if last_complete > 0:
                # Find the matching opening brace for the last complete dict
                brace_count = 0
                for i in range(last_complete, -1, -1):
                    if response[i] == '}':
                        brace_count += 1
                    elif response[i] == '{':
                        brace_count -= 1
                        if brace_count == 0:
                            response = response[:last_complete + 1] + ']'
                            break

        # Try Python eval
        try:
            result = eval(response)
            if isinstance(result, list):
                parsed = []
                for item in result:
                    if isinstance(item, dict) and 'query' in item and 'accepted' in item and 'rejected' in item:
                        parsed.append({
                            'query': str(item.get('query', '')),
                            'accepted': {
                                'think': str(item.get('accepted', {}).get('think', '')),
                                'answer': str(item.get('accepted', {}).get('answer', ''))
                            },
                            'rejected': {
                                'think': str(item.get('rejected', {}).get('think', '')),
                                'answer': str(item.get('rejected', {}).get('answer', ''))
                            }
                        })
                return parsed
        except:
            pass

        # Try JSON
        try:
            result = json.loads(response)
            if isinstance(result, list):
                parsed = []
                for item in result:
                    if isinstance(item, dict) and 'query' in item and 'accepted' in item and 'rejected' in item:
                        parsed.append({
                            'query': str(item.get('query', '')),
                            'accepted': {
                                'think': str(item.get('accepted', {}).get('think', '')),
                                'answer': str(item.get('accepted', {}).get('answer', ''))
                            },
                            'rejected': {
                                'think': str(item.get('rejected', {}).get('think', '')),
                                'answer': str(item.get('rejected', {}).get('answer', ''))
                            }
                        })
                return parsed
        except:
            pass

        return []

    def generate_single(
        self,
        prompt_creator: DMPOPromptCreator,
        num_interactions: int,
        conversations: int,
        target_tokens: int = 1024,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        timeout: int = 180,
        additional_config: dict = None,
        restart: bool = False,
        num_tries: int = 5
    ):
        """
        Generate DMPO pairs one interaction at a time.

        For DMPO, we build context using the ACCEPTED responses only,
        simulating how memory would be updated during actual training.

        Args:
            prompt_creator: DMPOPromptCreator instance
            num_interactions: Interactions per conversation
            conversations: Number of conversations to generate
            target_tokens: Target tokens per response
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens per generation
            timeout: Request timeout
            additional_config: Additional API config
            restart: Whether to clear existing items
            num_tries: Number of tries to generate the response
        """
        if restart:
            self.items = self._init_items()

        for conv_idx in range(conversations):
            topic = prompt_creator.get_random_topic()
            current_conversation = []

            # For DMPO, prior_interactions only include accepted responses
            # (simulating how memory is updated only with accepted during training)
            accepted_history = []

            for step in range(1, num_interactions + 1):
                system, user = prompt_creator.get_single_prompt(
                    topic=topic,
                    step_num=step,
                    total_steps=num_interactions,
                    prior_interactions=accepted_history if step > 1 else None,
                    target_tokens=target_tokens
                )
                for i in range(num_tries):
                    response = self.generate_items(
                        user,
                        stream=stream,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        system_prompt=system,
                        timeout=timeout,
                        additional_config=additional_config
                    )

                    if stream:
                        print('\n')

                    dmpo_pair = self._parse_dmpo_pair(response)
                    if dmpo_pair:
                        break
                    print(f"Failed to parse dmpo pair - {i+1} / {num_tries} attempts in interaction number {step+1} ")

                if dmpo_pair:
                    current_conversation.append(dmpo_pair)

                    # Update accepted history for next step
                    accepted_history.append({
                        'query': dmpo_pair['query'],
                        'think': dmpo_pair['accepted']['think'],
                        'answer': dmpo_pair['accepted']['answer']
                    })
                else:
                    print(f"Failed to parse DMPO pair {step} for conversation {conv_idx + 1}")
                    self.failed_count += 1
                    break

            # Only store complete conversations
            if len(current_conversation) == num_interactions:
                self.items['interactions'].append(current_conversation)
                self.items['topics'].append(topic)
                print(f"Generated DMPO conversation {conv_idx + 1}/{conversations} on '{topic[:50]}...'")
            else:
                print(f"Incomplete DMPO conversation {conv_idx + 1}, discarding")

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def generate_all_at_once(
        self,
        prompt_creator: DMPOPromptCreator,
        num_interactions: int,
        conversations: int,
        target_tokens_per_interaction: int = 1024,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 32000,
        timeout: int = 300,
        additional_config: dict = None,
        restart: bool = False
    ):
        """
        Generate entire DMPO conversations at once.

        Args:
            prompt_creator: DMPOPromptCreator instance
            num_interactions: Interactions per conversation
            conversations: Number of conversations to generate
            target_tokens_per_interaction: Target tokens per interaction
            stream: Whether to stream responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max tokens for entire generation
            timeout: Request timeout
            additional_config: Additional API config
            restart: Whether to clear existing items
        """
        if restart:
            self.items = self._init_items()

        for conv_idx in range(conversations):
            topic = prompt_creator.get_random_topic()

            system, user = prompt_creator.get_all_at_once_prompt(
                topic=topic,
                num_interactions=num_interactions,
                target_tokens_per_interaction=target_tokens_per_interaction
            )

            response = self.generate_items(
                user,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system_prompt=system,
                timeout=timeout,
                additional_config=additional_config
            )

            if stream:
                print('\n')

            conversation = self._parse_dmpo_conversation(response)

            if len(conversation) >= num_interactions * 0.8:  # Allow some tolerance
                # Pad or truncate to exact length
                while len(conversation) < num_interactions:
                    conversation.append({
                        'query': '',
                        'accepted': {'think': '', 'answer': ''},
                        'rejected': {'think': '', 'answer': ''}
                    })
                conversation = conversation[:num_interactions]

                self.items['interactions'].append(conversation)
                self.items['topics'].append(topic)
                print(f"Generated DMPO conversation {conv_idx + 1}/{conversations} on '{topic[:50]}...' ({len(conversation)} pairs)")
            else:
                print(f"Insufficient DMPO pairs ({len(conversation)}/{num_interactions}) for conversation {conv_idx + 1}")
                self.failed_count += 1

            if self.max_items and len(self.items['interactions']) >= self.max_items:
                print("Max items reached, stopping.")
                break

    def __call__(
        self,
        prompt_creator: DMPOPromptCreator,
        num_interactions: int,
        conversations: int,
        mode: Literal['single', 'all_at_once'] = 'single',
        **kwargs
    ):
        """
        Generate DMPO conversations.

        Args:
            prompt_creator: DMPOPromptCreator instance
            num_interactions: Interactions per conversation
            conversations: Number of conversations
            mode: 'single' or 'all_at_once'
            **kwargs: Additional arguments for the specific method
        """
        if mode == 'single':
            self.generate_single(
                prompt_creator=prompt_creator,
                num_interactions=num_interactions,
                conversations=conversations,
                **kwargs
            )
        elif mode == 'all_at_once':
            self.generate_all_at_once(
                prompt_creator=prompt_creator,
                num_interactions=num_interactions,
                conversations=conversations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'all_at_once'")


# ============================================================================
# POSTPROCESSORS
# ============================================================================

class HybridGeneratorPostprocessor:
    """
    Postprocessor for hybrid reasoning datasets.

    Provides utilities for:
    - Filtering duplicates
    - Validating conversation structure
    - Converting between formats
    - Pushing to HuggingFace Hub
    """

    def __init__(
        self,
        generator: Union[ReasoningCompletionGenerator, HybridReasoningGenerator, DMPOGenerator],
        dataset_id: str,
        config_name: str = None,
        split: str = 'train',
        token: str = None
    ):
        self.generator = generator
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.token = token

    def filter_empty_interactions(self):
        """Remove conversations with empty interactions."""
        filtered_interactions = []
        filtered_topics = []

        original_len = len(self.generator.items['interactions'])

        for i, conv in enumerate(self.generator.items['interactions']):
            # Check if all interactions have content
            valid = all(
                inter.get('query', '').strip() and inter.get('answer', '').strip()
                for inter in conv
            )

            if valid:
                filtered_interactions.append(conv)
                if 'topics' in self.generator.items and i < len(self.generator.items.get('topics', [])):
                    filtered_topics.append(self.generator.items['topics'][i])

        self.generator.items['interactions'] = filtered_interactions
        if 'topics' in self.generator.items:
            self.generator.items['topics'] = filtered_topics

        print(f"Filtered: {original_len} -> {len(filtered_interactions)} conversations")

    def filter_by_length(self, min_interactions: int = None, max_interactions: int = None):
        """Filter conversations by number of interactions."""
        filtered_interactions = []
        filtered_topics = []

        original_len = len(self.generator.items['interactions'])

        for i, conv in enumerate(self.generator.items['interactions']):
            conv_len = len(conv)

            if min_interactions and conv_len < min_interactions:
                continue
            if max_interactions and conv_len > max_interactions:
                continue

            filtered_interactions.append(conv)
            if 'topics' in self.generator.items and i < len(self.generator.items.get('topics', [])):
                filtered_topics.append(self.generator.items['topics'][i])

        self.generator.items['interactions'] = filtered_interactions
        if 'topics' in self.generator.items:
            self.generator.items['topics'] = filtered_topics

        print(f"Filtered by length: {original_len} -> {len(filtered_interactions)} conversations")

    def convert_to_rxt_format(self) -> list[dict]:
        """
        Convert conversations to RxT-Beta interaction template format.

        Returns list of conversations where each interaction is formatted as:
        - Fast answer: [Q] query [A] answer
        - Extended thinking: [Q] query [T] think [A] answer
        """
        rxt_conversations = []

        for conv in self.generator.items['interactions']:
            rxt_conv = []
            for inter in conv:
                query = inter.get('query', '')
                think = inter.get('think', '')
                answer = inter.get('answer', '')

                if think and len(think.strip()) > 10:
                    # Extended thinking mode
                    formatted = f"[Q] {query} [T] {think} [A] {answer}"
                else:
                    # Fast answer mode
                    formatted = f"[Q] {query} [A] {answer}"

                rxt_conv.append({
                    'formatted': formatted,
                    'query': query,
                    'think': think,
                    'answer': answer
                })

            rxt_conversations.append(rxt_conv)

        return rxt_conversations

    def push_to_hf_hub(self):
        """Push dataset to HuggingFace Hub."""
        ds = self.generator.get_dataset()
        if self.config_name is not None:
            ds.push_to_hub(
                repo_id=self.dataset_id,
                config_name=self.config_name,
                split=self.split,
                token=self.token
            )
        else:
            ds.push_to_hub(
                repo_id=self.dataset_id,
                split=self.split,
                token=self.token
            )
        print(f"Pushed to {self.dataset_id}")

    def append_from_existing_dataset(self):
        """Append items from an existing HuggingFace dataset."""
        if self.config_name is not None:
            dataset = load_dataset(
                self.dataset_id,
                self.config_name,
                split=self.split,
                token=self.token
            )
        else:
            dataset = load_dataset(
                self.dataset_id,
                split=self.split,
                token=self.token
            )

        # Merge existing with new
        for key in self.generator.items.keys():
            if key in dataset.column_names:
                self.generator.items[key] = list(dataset[key]) + self.generator.items[key]

        print(f"Merged with existing dataset, total: {len(self.generator.items['interactions'])}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_reasoning_completion_generator(
    max_items: int = 1000,
    model_name: str = "gpt-4",
    api_url: str = "https://api.openai.com/v1",
    api_key: str = None,
    use_ollama: bool = False
) -> ReasoningCompletionGenerator:
    """Create a reasoning completion generator with common defaults."""
    return ReasoningCompletionGenerator(
        max_items=max_items,
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        use_ollama=use_ollama
    )
import os
openai_api_key = os.environ.get("OPENAI_API_KEY")

# create_reasoning_completion_generator(model_name="gpt-4", use_ollama=False, max_items=1000, api_key=openai_api_key)

def create_hybrid_reasoning_generator(
    max_items: int = 1000,
    model_name: str = "gpt-4",
    api_url: str = "https://api.openai.com/v1",
    api_key: str = None,
    use_ollama: bool = False,
    topics: list[str] = None
) -> tuple[HybridReasoningGenerator, HybridReasoningPromptCreator]:
    """Create a hybrid reasoning generator with prompt creator."""
    generator = HybridReasoningGenerator(
        max_items=max_items,
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        use_ollama=use_ollama
    )
    prompt_creator = HybridReasoningPromptCreator(topics=topics)
    return generator, prompt_creator


def create_dmpo_generator(
    max_items: int = 1000,
    model_name: str = "gpt-4",
    api_url: str = "https://api.openai.com/v1",
    api_key: str = None,
    use_ollama: bool = False,
    topics: list[str] = None
) -> tuple[DMPOGenerator, DMPOPromptCreator]:
    """Create a DMPO generator with prompt creator."""
    generator = DMPOGenerator(
        max_items=max_items,
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        use_ollama=use_ollama
    )
    prompt_creator = DMPOPromptCreator(topics=topics)
    return generator, prompt_creator


