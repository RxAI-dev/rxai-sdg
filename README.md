# Reactive AI: Synthetic Dataset Generators (rxai-sdg)

Toolkit for generating high-quality synthetic datasets for training Reactive Transformer models. Supports Memory Reinforcement Learning (MRL), Supervised Fine-Tuning (SFT), and the new Hybrid Reasoning generators for RxT-Beta.

## Installation

```bash
pip install rxai-sdg
```

Or install from source:

```bash
git clone https://github.com/RxAI-dev/rxai-sdg.git
cd rxai-sdg
pip install -e .
```

## Overview

This library provides synthetic dataset generators for training Reactive Language Models:

| Module | Purpose | Target Training Stage |
|--------|---------|----------------------|
| `rxai_sdg.mrl` | Memory Reinforcement Learning datasets | MRL stage |
| `rxai_sdg.sft` | Supervised Fine-Tuning datasets | Interaction SFT |
| `rxai_sdg.hybrid` | Hybrid Reasoning & DMPO datasets | RxT-Beta advanced training |

## Quick Start

### API Configuration

All generators support both OpenAI-compatible APIs and Ollama for local testing:

```python
# OpenAI-compatible API (default)
generator = MrlSyntheticDatasetGenerator(
    max_items=100,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="your-api-key",
    use_ollama=False
)

# Ollama local testing
generator = MrlSyntheticDatasetGenerator(
    max_items=100,
    model_name="llama3.2",
    api_url="http://localhost:11434",
    use_ollama=True
)

# Third-party providers (Novita.ai, Together.ai, etc.)
generator = MrlSyntheticDatasetGenerator(
    max_items=100,
    model_name="qwen/qwen3-4b-fp8",
    api_url="https://api.novita.ai/v3/openai",
    api_key="your-key"
)
```

## Memory Reinforcement Learning (MRL) Datasets

Generate multi-turn conversations testing memory retention:

```python
from rxai_sdg.mrl import (
    MrlSyntheticDatasetGenerator,
    MrlPromptCreator,
    MrlGeneratorPostprocessor,
)
from rxai_sdg.mrl.prompts import ALL_PROMPTS_REAL, TOPICS_REAL
from rxai_sdg.mrl.examples import EXAMPLES_REAL_MICRO

# Initialize generator
generator = MrlSyntheticDatasetGenerator(
    max_items=500,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="your-key"
)

# Create prompt creator with topics
prompt_creator = MrlPromptCreator(
    prompts=ALL_PROMPTS_REAL,
    examples=EXAMPLES_REAL_MICRO,
    topics=TOPICS_REAL
)

# Generate dataset
generator(
    prompt_creator=prompt_creator,
    steps=3,           # Follow-up interactions per conversation
    iterations=50,     # API calls
    num_examples=10,   # Examples per API call
    num_topics=10,     # Random topics per prompt
    temperature=0.7,
    stream=True,       # Show generation progress
    max_tokens=20000
)

# Post-process and export
postprocessor = MrlGeneratorPostprocessor(
    generator=generator,
    dataset_id="your-org/mrl-dataset",
    token="hf_token"
)
postprocessor.filter_duplicates()
postprocessor.remove_incorrect_interactions(steps=3)
postprocessor.push_to_hf_hub()

# Or get as Dataset object
dataset = generator.get_dataset()
```

### MRL Dataset Format

```python
{
    'query': ["Initial question 1", "Initial question 2", ...],
    'answer': ["Initial answer 1", "Initial answer 2", ...],
    'interactions': [
        [
            {'query': "Follow-up Q1", 'answer': "Follow-up A1"},
            {'query': "Follow-up Q2", 'answer': "Follow-up A2"},
            ...
        ],
        ...
    ]
}
```

### Generation Modes

**Multi-Topic Mode** (default): Single topic with progressive memory testing
```python
generator(prompt_creator, steps=3, mode='multi')
```

**Long-Range Mode**: Two-topic strategy testing long-range memory
```python
generator(prompt_creator, steps=5, mode='long')
```

## RxT-Beta Hybrid Reasoning Datasets

The `rxai_sdg.hybrid` module provides generators for RxT-Beta's advanced training stages:

### 1. Reasoning Completion Generator

Add missing 'think' blocks to existing conversations:

```python
from rxai_sdg.hybrid import ReasoningCompletionGenerator
from datasets import load_dataset

# Load existing dataset with missing think blocks
dataset = load_dataset("your-dataset", split="train")
# Expected format: {'interactions': [[{'query': ..., 'think': '', 'answer': ...}, ...]]}

generator = ReasoningCompletionGenerator(
    max_items=100,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="your-key"
)

# Mode 1: Generate think blocks one at a time (higher quality)
generator.complete_single(
    dataset=dataset,
    target_tokens=512,
    temperature=0.7,
    stream=True
)

# Mode 2: Generate all think blocks at once (more efficient)
generator.complete_all_at_once(
    dataset=dataset,
    target_tokens_per_think=512,
    temperature=0.7
)

# Get completed dataset
completed_dataset = generator.get_dataset()
```

### 2. Hybrid Reasoning Generator

Create new conversations with full reasoning chains from scratch:

```python
from rxai_sdg.hybrid import (
    HybridReasoningGenerator,
    HybridReasoningPromptCreator,
    TOPICS_HYBRID_REASONING,
)

# Initialize
generator = HybridReasoningGenerator(
    max_items=100,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="your-key"
)

# Custom topics (or use defaults)
my_topics = [
    "Quantum computing fundamentals",
    "Climate change feedback loops",
    "Machine learning optimization",
    # ...
]

prompt_creator = HybridReasoningPromptCreator(
    topics=my_topics,  # or TOPICS_HYBRID_REASONING
    include_examples=True
)

# Mode 1: Generate one interaction at a time (builds context progressively)
generator.generate_single(
    prompt_creator=prompt_creator,
    num_interactions=5,    # Interactions per conversation
    conversations=20,      # Number of conversations
    target_tokens=1024,    # Tokens per interaction
    thinking_ratio=0.7,    # 70% use extended thinking
    temperature=0.7,
    stream=True
)

# Mode 2: Generate entire conversations at once
generator.generate_all_at_once(
    prompt_creator=prompt_creator,
    num_interactions=5,
    conversations=20,
    target_tokens_per_interaction=1024,
    thinking_ratio=0.7,
    temperature=0.7
)

dataset = generator.get_dataset()
```

### Hybrid Reasoning Dataset Format

```python
{
    'interactions': [
        [
            {'query': "Question 1", 'think': "Reasoning...", 'answer': "Response 1"},
            {'query': "Question 2", 'think': "Reasoning...", 'answer': "Response 2"},
            ...
        ],
        ...
    ],
    'topics': ["Topic 1", "Topic 2", ...]
}
```

### 3. DMPO (Direct Memory and Preference Optimization) Generator

Create preference pairs for memory-aware training:

```python
from rxai_sdg.hybrid import DMPOGenerator, DMPOPromptCreator

generator = DMPOGenerator(
    max_items=100,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="your-key"
)

prompt_creator = DMPOPromptCreator(
    topics=TOPICS_HYBRID_REASONING,
    include_examples=True
)

# Mode 1: Generate pairs one at a time
generator.generate_single(
    prompt_creator=prompt_creator,
    num_interactions=5,
    conversations=20,
    target_tokens=1024,
    temperature=0.7
)

# Mode 2: Generate entire preference conversations at once
generator.generate_all_at_once(
    prompt_creator=prompt_creator,
    num_interactions=5,
    conversations=20,
    target_tokens_per_interaction=1024
)

dataset = generator.get_dataset()
```

### DMPO Dataset Format

Each interaction contains accepted (good) and rejected (bad) responses:

```python
{
    'interactions': [
        [
            {
                'query': "Question requiring memory...",
                'accepted': {
                    'think': "Good reasoning with correct memory usage...",
                    'answer': "Accurate, helpful response..."
                },
                'rejected': {
                    'think': "Flawed reasoning or memory errors...",
                    'answer': "Response with issues..."
                }
            },
            ...
        ],
        ...
    ],
    'topics': ["Topic 1", "Topic 2", ...]
}
```

### Postprocessing

```python
from rxai_sdg.hybrid import HybridGeneratorPostprocessor

postprocessor = HybridGeneratorPostprocessor(
    generator=generator,
    dataset_id="your-org/hybrid-dataset",
    token="hf_token"
)

# Filter empty/invalid conversations
postprocessor.filter_empty_interactions()

# Filter by conversation length
postprocessor.filter_by_length(min_interactions=3, max_interactions=10)

# Convert to RxT-Beta format
rxt_format = postprocessor.convert_to_rxt_format()
# Returns: [{'formatted': '[Q] query [T] think [A] answer', ...}, ...]

# Push to HuggingFace Hub
postprocessor.push_to_hf_hub()
```

## RxT-Beta Interaction Template

The hybrid generators produce data compatible with RxT-Beta's interaction template:

| Mode | Template | Description |
|------|----------|-------------|
| Fast Answer | `[Q] query [A] answer` | Direct response without reasoning |
| Extended Thinking | `[Q] query [T] thinking [A] answer` | Response with reasoning chain |
| Tool Usage | `[U] tool_result [T] thinking [A] answer` | Processing tool results |
| Internal Instruction | `[I] instruction [Q] query [A] answer` | Per-interaction behavior control |
| Tool Call | `[Q] query [A] answer [C] tool_call` | Invoking external tools |

## Convenience Functions

```python
from rxai_sdg.hybrid import (
    create_reasoning_completion_generator,
    create_hybrid_reasoning_generator,
    create_dmpo_generator,
)

# Quick setup with defaults
completion_gen = create_reasoning_completion_generator(
    max_items=100,
    model_name="gpt-4",
    api_key="your-key"
)

# Returns (generator, prompt_creator) tuple
reasoning_gen, reasoning_prompts = create_hybrid_reasoning_generator(
    max_items=100,
    model_name="gpt-4",
    api_key="your-key",
    topics=my_custom_topics
)

dmpo_gen, dmpo_prompts = create_dmpo_generator(
    max_items=100,
    model_name="gpt-4",
    api_key="your-key"
)
```

## API Reference

### Base Classes

#### `BaseDatasetGenerator`

Abstract base class for all generators.

```python
class BaseDatasetGenerator(ABC):
    def __init__(
        self,
        max_items: int = None,           # Maximum items to generate
        model_name: str = "...",          # Model identifier
        api_url: str = "...",             # API endpoint
        api_key: str = None,              # API authentication
        use_ollama: bool = False          # Use Ollama instead of OpenAI API
    )

    def generate_items(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 15000,
        system_prompt: str = "",
        timeout: int = 120,
        additional_config: dict = None
    ) -> str

    def get_dataset(self) -> Dataset  # Return HuggingFace Dataset
```

### MRL Module

- `MrlSyntheticDatasetGenerator` - Main generator
- `MrlPromptCreator` - Prompt composition
- `MrlContextBasedPromptCreator` - Context-aware prompts
- `MrlGeneratorPostprocessor` - Post-processing and export

### Hybrid Module

- `ReasoningCompletionGenerator` - Add missing think blocks
- `HybridReasoningGenerator` - Create reasoning conversations
- `DMPOGenerator` - Create preference pairs
- `HybridReasoningPromptCreator` - Prompts for reasoning generation
- `DMPOPromptCreator` - Prompts for DMPO generation
- `HybridGeneratorPostprocessor` - Post-processing utilities

## Configuration Options

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Sampling temperature (0-1) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `max_tokens` | 15000 | Maximum tokens per generation |
| `timeout` | 120 | Request timeout in seconds |
| `stream` | False | Stream responses in real-time |

### Additional Config

```python
additional_config = {
    'presence_penalty': 0,
    'frequency_penalty': 0,
    'response_format': {"type": "text"},
    'extra_body': {
        "top_k": 50,
        'repetition_penalty': 1,
        'min_p': 0,
    },
}

generator.generate_items(..., additional_config=additional_config)
```

## Examples

### Complete Workflow: MRL Dataset

```python
from rxai_sdg.mrl import *
from rxai_sdg.mrl.prompts import ALL_PROMPTS_REAL, TOPICS_REAL

# 1. Initialize
generator = MrlSyntheticDatasetGenerator(
    max_items=1000,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="sk-..."
)

prompt_creator = MrlPromptCreator(
    prompts=ALL_PROMPTS_REAL,
    topics=TOPICS_REAL
)

# 2. Generate
for steps in [2, 3, 4, 5]:  # Multiple conversation lengths
    generator(
        prompt_creator=prompt_creator,
        steps=steps,
        iterations=25,
        num_examples=10,
        temperature=0.8,
        stream=True
    )

# 3. Post-process
postprocessor = MrlGeneratorPostprocessor(
    generator=generator,
    dataset_id="myorg/mrl-dataset",
    token="hf_..."
)
postprocessor.filter_duplicates()
postprocessor.push_to_hf_hub()
```

### Complete Workflow: Hybrid Reasoning

```python
from rxai_sdg.hybrid import *

# 1. Initialize
generator = HybridReasoningGenerator(
    max_items=500,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="sk-..."
)

prompt_creator = HybridReasoningPromptCreator()

# 2. Generate different conversation lengths
for length in [3, 5, 7]:
    generator(
        prompt_creator=prompt_creator,
        num_interactions=length,
        conversations=50,
        mode='single',  # Higher quality
        temperature=0.7,
        stream=True,
        restart=False   # Accumulate
    )

# 3. Post-process
postprocessor = HybridGeneratorPostprocessor(
    generator=generator,
    dataset_id="myorg/hybrid-reasoning",
    token="hf_..."
)
postprocessor.filter_empty_interactions()
postprocessor.push_to_hf_hub()
```

### Complete Workflow: DMPO Dataset

```python
from rxai_sdg.hybrid import *

# 1. Initialize
generator = DMPOGenerator(
    max_items=300,
    model_name="gpt-4",
    api_url="https://api.openai.com/v1",
    api_key="sk-..."
)

prompt_creator = DMPOPromptCreator()

# 2. Generate
generator(
    prompt_creator=prompt_creator,
    num_interactions=5,
    conversations=60,
    mode='single',
    target_tokens=1024,
    temperature=0.7
)

# 3. Export
postprocessor = HybridGeneratorPostprocessor(
    generator=generator,
    dataset_id="myorg/dmpo-dataset",
    token="hf_..."
)
postprocessor.push_to_hf_hub()
```

## Ollama Local Testing

For local development and testing with Ollama:

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2
```

```python
from rxai_sdg.hybrid import HybridReasoningGenerator, HybridReasoningPromptCreator

# Use Ollama
generator = HybridReasoningGenerator(
    max_items=10,
    model_name="llama3.2",
    api_url="http://localhost:11434",
    use_ollama=True
)

prompt_creator = HybridReasoningPromptCreator()

# Generate (smaller batches for local testing)
generator(
    prompt_creator=prompt_creator,
    num_interactions=3,
    conversations=5,
    mode='single',
    temperature=0.8,
    stream=True
)
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Links

- [Repository](https://github.com/RxAI-dev/rxai-sdg)
- [RxT-Beta Model](https://huggingface.co/ReactiveAI/RxT-Beta)
- [Reactive Transformer Paper](https://arxiv.org/abs/2510.03561)
