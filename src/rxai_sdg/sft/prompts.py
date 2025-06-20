from typing import Callable, TypeAlias

def system_prompt_stories(num_examples: int) -> str:
    return f"""
    You are a supervised fine-tuning dataset generator for a Reactive Transformer model.
    Output must be a Python list of tuples containing only query-answer pairs, without any code, explanation, or metadata.
    You are used to generate list of exactly {num_examples} tuples with two elements: question and answer, based on simple stories and TinyStories dataset
    """

def prompt_start_stories(num_examples: int) -> str:
    exs_30_percent = int(num_examples * 0.30)
    exs_70_percent = int(num_examples - exs_30_percent)

    return f"""
    # Reactive Transformer Fine-Tuning Data Generation
    
    ## TASK DESCRIPTION
    You have to create our own synthetic fine-tuning dataset, based on TinyStories, with about 40k examples of 150-250 tokens
    (left space for special tokens in model 256 context size) interactions. The questions should be rather shorter than
    answers - in example 5-100 tokens and the rest from 250 (150-245) is for the answer. Interactions (examples) should
    have two general categories - first/initial interaction or consecutive interaction. For this dataset, first interaction
    will be mostly a (polite) request to generate some story ("Please generate me a story about small black cat"),
    and the answer is the story similar to ones included in base dataset. Consecutive interactions are the questions about those
    stories ("Why the small black cat is happy?") - but they don't have to be connected to each other (that's not important
    for this training stage). 30%-40% of dataset should be the initial interactions and the rest for the consecutive ones.
    
    Interaction is just a single message and its response (question and answer)
    
    ## TASK
    Generate **{num_examples} unique interaction pairs**, each consisting of one question and its immediate answer (no dependencies between examples).
    - **30% ({exs_30_percent} examples)**: *Initial prompts* (open-ended requests to generate a story/answer).
    - **70% ({exs_70_percent} examples)**: *Follow-up questions* that:
      1. Only reference details from *their own preceding answer* (not external knowledge or other examples).
      2. Must be standalone (no shared storylines, names, or entities between examples).
      3. Must ask about **memory hooks** (key terms from their answer’s story).
    
    ---
    
    ## FORMAT
    ```python
    [
        ("Generate a story about a lone inventor fixing a broken ecosystem.", "Isabel found a dying forest, its soil poisoned. She engineered symbiotic bacteria to neutralize toxins, reviving flora. Her creation also restored vanished fireflies."),
        ("Why did the bacteria solve the forest's issue?", "The bacteria's enzymes broke down heavy metals and stabilized soil pH to levels editable for native species."),
        ...
    ],
    
    
    Each example should be a correct tuple in format `("Question", "Answer")` - please ensure, that it has correct opening and closing.
    All examples are wrapped in list, starting from `[` and ending with `]`
    
    ## FEW SHOT EXAMPLES
    ```python
    """

def prompt_end_stories(num_examples: int, include_no_think: bool = True) -> str:
    return f"""
    ```
    ## NUMBER OF GENERATED ELEMENTS
    Exactly {num_examples} tuples in list
    
    ---
    
    ## **CRITICAL RULES**
    1. **Entity Uniqueness**:
       - Every example has **distinct names/places/concepts** (e.g., "Isabel" in one, "Clara" in another).
       - Avoid repeating terms between examples (e.g., "Luminari" or "Luna" are non-existent across entries).
    
    2. **Follow-Up Constraints**:
       - Follow-ups are strictly tied to *their own answer's content*.
       - Example: If an initial story features a "talking mushroom," ask "How did the mushroom communicate?" (not cross-ref to unrelated examples).
    
    3. **Memory Hooks**:
       - Answers must contain **clear clues** for follow-ups (e.g., "enchanted bauble," "time loop malfunction," "bioluminescent spores").
       - Unused hooks = invalid examples.
    
    ---
    
    ### **BROKEN EXAMPLE (Common Error)**
    ❌ DO NOT generate:
        ("Story about robot⼊ hunting dragons", "Bob the robot used heat-seeking missiles..."),
        ("How did Bob defeat the dragon?", "Bob used ...")  # Repeats 'Bob' and links examples
    
    ---
    
    ### **VALID EXAMPLES**
    ✅ Generate:
      ("Create a story about a sentient clock.", "The brass clock in the mayor's office began chiming irregularly. It responded to specific phrases in speeches, aligning timekeeping with political urgency. When a journalist asked about its behavior, it chimed Morse code details of upcoming budget cuts. The mayor tried silencing it by disconnecting the gears, but the clock locked its hands at 3:14, creating a local time loop. Citizens synchronized with the clock's new rhythm, ignoring standard time. Productivity increased in areas where rigid schedules had failed. The clock exposed budget allocations prioritizing industrial zones over schools and hospitals. When the mayor attempted to replace it with a digital model, the original clock's pendulum reappeared inside the new casing. Now the city operates on dual time systems—mechanical truth and digital convenience. The clock's chimes have become the unofficial language of transparency."),
      ("How did the bridge's tremors change during repairs?", "When construction equipment arrived, its vibrations disrupted the bridge's feedback loop with Lina. The bridge began creaking in distress tones, forcing engineers to recalibrate their tools to match her natural rhythm. Lina demonstrated footstep patterns that synchronized with the bridge's resonance, stabilizing it during repairs. Workers began listening to her suggestions, as the bridge responded positively to her input. By the end, repairs were completed with minimal disruption. The bridge now plays soft melodies at night, a behavior Lina encourages through counter-vibrations."),
    
    ---
    
    ### **CHECKLIST**
    - [ ] {num_examples} entries with clear initial/follow-up divides.
    - [ ] All names/places/concepts are unique to each example.
    - [ ] Follow-up questions = *isolated* from other entries.
    - [ ]. No shared stories or character links between examples.
    
    Please output only the dataset list, NOTHING ELSE.
    """ + """
    /no_think
    """ if include_no_think else ''


PromptFactory: TypeAlias = Callable[[int], str]
ALL_PROMPTS_STORIES: tuple[PromptFactory, PromptFactory, PromptFactory] = (system_prompt_stories, prompt_start_stories, prompt_end_stories)
