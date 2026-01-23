"""
Prompts for RxT-Beta Hybrid Reasoning Dataset Generators

These prompts are used for:
1. Reasoning Completion - Adding missing 'think' blocks to existing conversations
2. Reasoning Dataset Generation - Creating new conversations with hybrid reasoning
3. DMPO Dataset Generation - Creating preference pairs for memory optimization

RxT-Beta Interaction Template:
- [Q] query [A] answer - Fast answer mode
- [Q] query [T] thinking [A] answer - Extended thinking mode
- [U] tool_result [T] thinking [A] answer - Tool usage mode
- [I] instruction - Internal instruction (before query)
- [C] tool_call - Agentic tool call (after answer)
"""

import random
from typing import Literal

# Topics for generating diverse hybrid reasoning conversations
TOPICS_HYBRID_REASONING = [
    # Scientific reasoning
    "Quantum entanglement and its applications in computing",
    "CRISPR gene editing ethical considerations",
    "Black hole information paradox",
    "Protein folding prediction mechanisms",
    "Climate change feedback loops",
    "Evolutionary advantages of cooperation",
    "Neuroplasticity and learning mechanisms",
    "Dark matter detection methods",
    "Antibiotic resistance evolution",
    "Photosynthesis efficiency optimization",

    # Mathematical reasoning
    "Prime number distribution patterns",
    "Non-Euclidean geometry applications",
    "Chaos theory and weather prediction",
    "Graph theory in social networks",
    "Optimization algorithms comparison",
    "Cryptographic hash function security",
    "Statistical paradoxes (Simpson's, Berkson's)",
    "Game theory equilibrium strategies",
    "Fractal dimension calculations",
    "Probability in quantum mechanics",

    # Logical puzzles and problems
    "Knights and knaves logic puzzles",
    "Monty Hall problem variations",
    "River crossing puzzles with constraints",
    "Scheduling optimization problems",
    "Resource allocation dilemmas",
    "Truth-teller paradoxes",
    "Constraint satisfaction problems",
    "Multi-agent coordination challenges",
    "Decision trees for diagnosis",
    "Causal reasoning chains",

    # Programming and algorithms
    "Recursive vs iterative solutions tradeoffs",
    "Time-space complexity analysis",
    "Dynamic programming optimization",
    "Parallel algorithm design",
    "Data structure selection criteria",
    "API design best practices",
    "Concurrency and deadlock prevention",
    "Memory management strategies",
    "Compiler optimization techniques",
    "Distributed systems consistency",

    # Real-world analysis
    "Supply chain optimization strategies",
    "Traffic flow network analysis",
    "Epidemic spread modeling",
    "Financial risk assessment models",
    "Energy grid load balancing",
    "Urban planning considerations",
    "Healthcare resource allocation",
    "Environmental impact assessment",
    "Educational curriculum design",
    "Agricultural yield optimization",

    # Philosophy and ethics
    "Trolley problem variations",
    "AI alignment challenges",
    "Free will and determinism debate",
    "Moral relativism vs universalism",
    "Rights of future generations",
    "Privacy vs security tradeoffs",
    "Intellectual property ethics",
    "Animal consciousness evidence",
    "Personal identity persistence",
    "Collective vs individual rights",

    # Historical analysis
    "Causes of ancient civilization collapses",
    "Technology transfer in history",
    "Economic factors in revolutions",
    "Disease impacts on historical events",
    "Cultural exchange effects",
    "Military strategy evolution",
    "Scientific revolution catalysts",
    "Trade route development impacts",
    "Institutional development patterns",
    "Migration and cultural change",

    # Technology assessment
    "AI safety research priorities",
    "Renewable energy transition challenges",
    "Space exploration cost-benefit analysis",
    "Autonomous vehicle ethics",
    "Social media impact assessment",
    "Biotechnology regulation needs",
    "Cybersecurity threat evolution",
    "Digital currency implications",
    "Quantum computing timeline",
    "Brain-computer interface ethics",

    # Creative problem solving
    "Innovation in constrained environments",
    "Cross-domain knowledge transfer",
    "Analogical reasoning applications",
    "Lateral thinking exercises",
    "Reverse engineering approaches",
    "Biomimicry design solutions",
    "Emergent behavior prediction",
    "System dynamics modeling",
    "Scenario planning methods",
    "Design thinking process",

    # Multi-step reasoning chains
    "Chain of custody in forensics",
    "Diagnostic reasoning in medicine",
    "Legal argument construction",
    "Scientific hypothesis testing",
    "Debugging complex systems",
    "Root cause analysis methods",
    "Strategic planning sequences",
    "Proof construction in mathematics",
    "Policy impact prediction",
    "Engineering failure analysis",
]


# ============================================================================
# REASONING COMPLETION PROMPTS
# For adding missing 'think' blocks to existing conversations
# ============================================================================

def system_reasoning_completion_single(model_context: str = "RxT-Beta"):
    """System prompt for single think block generation."""
    return f"""You are a reasoning completion generator for {model_context}, a Reactive Language Model with hybrid reasoning capabilities.

Your task is to generate high-quality reasoning/thinking blocks that would naturally precede given answers in a multi-turn conversation.

The reasoning should:
- Show step-by-step thought process leading to the answer
- Reference relevant information from previous interactions (memory context)
- Demonstrate logical connections and inferences
- Be coherent with both the query and the provided answer
- Match the complexity level of the question

Output format: Generate ONLY the thinking content, without any special tokens or markers.
The thinking block should be self-contained reasoning text."""


def system_reasoning_completion_all():
    """System prompt for generating all think blocks at once."""
    return """You are a reasoning completion generator for RxT-Beta, a Reactive Language Model with hybrid reasoning capabilities.

Your task is to generate high-quality reasoning/thinking blocks for ALL interactions in a conversation at once.

For each interaction, the reasoning should:
- Show step-by-step thought process leading to the answer
- Reference relevant information from previous interactions when applicable
- Demonstrate logical connections and inferences
- Be coherent with both the query and the provided answer
- Build upon previous reasoning where appropriate

NSTRUCTION FOR MULTILINE RESPONSES:

    Always use the escape sequence \\n to indicate a line break.

    Never use actual newline characters (from pressing Enter/Return).

    Never use HTML's
    or Windows-style \\r\\n.

    Example of correct formatting: "First line\\nSecond line\\nThird line"

    This ensures your response can be parsed correctly by the system.
Output format: A Python list of strings, where each string is the thinking block for the corresponding interaction.
Example: ["First thinking block...", "Second thinking block...", ...]"""


def task_description_reasoning_completion_single(
    query: str,
    answer: str,
    memory_context: list[dict] = None,
    target_tokens: int = 512
):
    """Generate prompt for single think block completion."""
    memory_str = ""
    if memory_context:
        memory_str = "\n## PREVIOUS INTERACTIONS (Memory Context)\n"
        for i, ctx in enumerate(memory_context, 1):
            memory_str += f"\n### Interaction {i}\n"
            memory_str += f"Query: {ctx.get('query', '')}\n"
            if ctx.get('think'):
                memory_str += f"Thinking: {ctx.get('think', '')}\n"
            memory_str += f"Answer: {ctx.get('answer', '')}\n"

    return f"""# Reasoning Completion Task

## TASK
Generate the thinking/reasoning block that would naturally lead from the query to the given answer.
{memory_str}
## CURRENT INTERACTION
Query: {query}

Answer: {answer}

## REQUIREMENTS
1. The thinking should logically connect the query to the answer
2. Reference specific facts from memory context if relevant
3. Show clear reasoning steps (analysis, consideration of factors, conclusions)
4. Target approximately {target_tokens} tokens
5. Do not repeat the answer in the thinking - focus on the reasoning process
6. If the query requires combining information from multiple previous interactions, explicitly reference them

## OUTPUT
Generate ONLY the thinking content - no special tokens, no explanation, just the reasoning text."""


def task_description_reasoning_completion_all(
    interactions: list[dict],
    target_tokens_per_think: int = 512
):
    """Generate prompt for all think blocks at once."""
    interactions_str = ''''''
    for i, inter in enumerate(interactions, 1):
        interactions_str += f"\n### Interaction {i}\n"
        interactions_str += f"Query: {inter.get('query', '')}\n"
        interactions_str += f"Answer: {inter.get('answer', '')}\n"

    return f"""# Multi-Interaction Reasoning Completion Task

    ## TASK
    Generate thinking/reasoning blocks for ALL interactions in this conversation at once.
    Each thinking block should logically connect its query to its answer.

    ## CONVERSATION
    {interactions_str}
    
    ## REQUIREMENTS
    1. Generate a thinking block for EACH interaction
    2. Later interactions should reference earlier ones when relevant
    3. Each thinking should show clear reasoning steps leading to its answer
    4. Target approximately {target_tokens_per_think} tokens per thinking block
    5. Do not repeat answers in thinking - focus on reasoning process
    6. Maintain coherence across the conversation
    
    ## OUTPUT FORMAT
    Output a Python list of strings where each string is a thinking block:
    ["Thinking for interaction 1...", "Thinking for interaction 2...", ...]
    
    Generate ONLY the Python list - no other text or explanation."""


# ============================================================================
# REASONING DATASET GENERATION PROMPTS
# For creating new conversations with hybrid reasoning from scratch
# ============================================================================

def system_reasoning_generation_single():
    """System prompt for single interaction generation."""
    return """You are a Hybrid Reasoning Dataset Generator for RxT-Beta, a Reactive Language Model.

Your task is to generate high-quality single interactions with full reasoning chains.

Each interaction consists of:
- Query: A question or task requiring reasoning
- Think: Step-by-step reasoning process
- Answer: The final response

The interaction should:
- Require genuine multi-step reasoning
- Reference previous context when provided
- Show clear logical progression
- Be factually accurate where applicable
- Demonstrate memory-aware reasoning when building on prior interactions

- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar).
All reasoning must strictly relate to the subject matter of the query.

Output must follow the exact format specified.
"""


def system_reasoning_generation_all():
    """System prompt for full conversation generation."""
    return """You are a Hybrid Reasoning Dataset Generator for RxT-Beta, a Reactive Language Model with infinite memory.

Your task is to generate complete multi-turn conversations with full reasoning chains.

Each conversation should:
- Have interconnected interactions testing memory and reasoning
- Build progressively on previous information
- Require combining facts across multiple turns
- Demonstrate both fast answers (simple) and extended thinking (complex)
- Show realistic conversational flow


Output must be a Python list in the exact format specified."""


def task_description_reasoning_generation_single(
    topic: str,
    step_num: int,
    total_steps: int,
    prior_interactions: list[dict] = None,
    target_tokens: int = 1024,
    require_extended_thinking: bool = True
):
    """Generate prompt for creating a single interaction."""
    prior_str = ""
    if prior_interactions:
        prior_str = "\n## PRIOR INTERACTIONS (Must reference these)\n"
        for i, inter in enumerate(prior_interactions, 1):
            prior_str += f"\n### Interaction {i}\n"
            prior_str += f"Query: {inter.get('query', '')}\n"
            if inter.get('think'):
                prior_str += f"Think: {inter.get('think', '')}\n"
            prior_str += f"Answer: {inter.get('answer', '')}\n"

    thinking_requirement = """
## THINKING REQUIREMENTS
- Must show clear step-by-step reasoning
- Reference specific facts from prior interactions
- Consider multiple aspects before concluding
- Target {target_tokens} tokens for think + answer combined""" if require_extended_thinking else """
- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar).
All reasoning must strictly relate to the subject matter of the query. Below there is an example of incorrect content inside "think" block:
    - "I also need to keep the response within roughly 350 words to stay near the 1024‑token target when combined with the answer"

## ANSWER MODE
- This is a simpler question that can be answered directly
- Thinking can be brief or omitted
- Focus on providing accurate, helpful response"""

    return f"""# Single Interaction Generation - Step {step_num}/{total_steps}

## TOPIC
{topic}
{prior_str}
## TASK
Generate interaction {step_num} of a {total_steps}-step conversation.
{"This should reference and build upon prior interactions." if prior_interactions else "This is the first interaction - establish the topic."}
{thinking_requirement.format(target_tokens=target_tokens)}

## OUTPUT FORMAT
Output a Python dictionary with exactly these keys:
{{"query": 'The question...', "think": 'Step-by-step reasoning...', "answer": 'Final response...'}}
Use \\n as a newline instead of actual newline <enter> - the text must be easily evaluated by python's eval() function 
Generate ONLY the dictionary - no other text."""


def task_description_reasoning_generation_all(
    topic: str,
    num_interactions: int,
    target_tokens_per_interaction: int = 1024,
    thinking_ratio: float = 0.7
):
    """Generate prompt for creating a full conversation at once."""
    num_thinking = max(1, int(num_interactions * thinking_ratio))
    num_fast = num_interactions - num_thinking

    return f"""# Full Conversation Generation

## TOPIC
{topic}

## TASK
Generate a complete {num_interactions}-interaction conversation demonstrating hybrid reasoning.

## REQUIREMENTS
1. Conversation should have {num_interactions} interconnected interactions
2. Approximately {num_thinking} interactions should use extended thinking (complex questions)
3. Approximately {num_fast} interactions can be fast answers (simpler follow-ups)
4. Each interaction should reference information from previous ones
5. Final interaction should synthesize information from multiple prior steps
6. Target approximately {target_tokens_per_interaction} tokens per interaction (query + think + answer)

## INTERACTION STRUCTURE
- Early interactions: Establish facts and context
- Middle interactions: Build complexity, combine information
- Later interactions: Require synthesis across multiple steps

## MEMORY TESTING
Ensure the conversation tests memory retention by:
- Referencing specific facts from earlier interactions
- Requiring combination of information across turns
- Using progressive knowledge building

## OUTPUT FORMAT
Output a Python list of dictionaries:
[
    {{"query": "...", "think": "...", "answer": "..."}},
    {{"query": "...", "think": "...", "answer": "..."}},
    # ... {num_interactions} total interactions
]

For fast-answer interactions, 'think' can be empty string "".


Generate ONLY the Python list - no other text."""


# ============================================================================
# DMPO (Direct Memory and Preference Optimization) PROMPTS
# For creating preference pairs with accepted/rejected responses
# ============================================================================

def system_dmpo_generation_single():
    """System prompt for single DMPO pair generation."""
    return """You are a DMPO Dataset Generator for RxT-Beta, creating preference pairs for memory optimization training.

Your task is to generate BOTH an accepted (good) and rejected (bad) response for each interaction.

ACCEPTED response characteristics:
- Excellent memory usage - accurately references previous interactions
- High-quality reasoning with clear logical steps
- Accurate, helpful, and comprehensive answer
- Appropriate use of extended thinking when needed

REJECTED response characteristics:
- Poor memory usage - misses or misremembers previous information
- Weak or flawed reasoning
- Incomplete, inaccurate, or unhelpful answer
- May contain subtle errors or contradictions

The contrast should be clear but realistic - rejected responses should represent plausible mistakes, not absurd errors."""


def system_dmpo_generation_all():
    """System prompt for full DMPO conversation generation."""
    return """You are a DMPO Dataset Generator for RxT-Beta, creating complete preference-paired conversations.

For each interaction in a conversation, you generate BOTH:
- Accepted (good) response: Excellent memory usage, strong reasoning, accurate answers
- Rejected (bad) response: Poor memory usage, weak reasoning, or errors

This training data teaches the model to:
1. Properly utilize information stored in memory
2. Produce high-quality reasoning chains
3. Give accurate, helpful responses

Make rejected responses realistic - they should represent plausible errors a model might make, not obviously wrong answers."""


def task_description_dmpo_single(
    topic: str,
    step_num: int,
    total_steps: int,
    prior_interactions: list[dict] = None,
    target_tokens: int = 1024
):
    """Generate prompt for creating a single DMPO preference pair."""
    prior_str = ""
    if prior_interactions:
        prior_str = "\n## PRIOR INTERACTIONS (Memory Context)\n"
        for i, inter in enumerate(prior_interactions, 1):
            prior_str += f"\n### Interaction {i}\n"
            prior_str += f"Query: {inter.get('query', '')}\n"
            if inter.get('think'):
                prior_str += f"Think: {inter.get('think', '')}\n"
            prior_str += f"Answer: {inter.get('answer', '')}\n"

    return f"""# DMPO Pair Generation - Step {step_num}/{total_steps}

## TOPIC
{topic}
{prior_str}
## TASK
Generate interaction {step_num} with BOTH accepted and rejected responses.

## ACCEPTED RESPONSE REQUIREMENTS
- Accurately reference facts from prior interactions
- Clear, logical reasoning in 'think' block
- Comprehensive, helpful answer
- No factual errors or contradictions

## REJECTED RESPONSE REQUIREMENTS (Choose 1-2 issues)
- Memory errors: Misremember facts, miss relevant prior information
- Reasoning errors: Flawed logic, missing steps, wrong conclusions
- Answer quality: Incomplete, slightly wrong, or unhelpful
- Make it realistic - a plausible mistake, not obviously absurd

## OUTPUT FORMAT
Output a Python dictionary with this structure:
{{
    "query": "The question requiring memory and reasoning...",
    "accepted": {{
        "think": "Good step-by-step reasoning referencing memory...",
        "answer": "Accurate, helpful response..."
    }},
    "rejected": {{
        "think": "Flawed reasoning or missing memory references...",
        "answer": "Response with subtle issues..."
    }}
}}

Target approximately {target_tokens} tokens total per response (think + answer).

Generate ONLY the dictionary - no other text."""


def task_description_dmpo_all(
    topic: str,
    num_interactions: int,
    target_tokens_per_interaction: int = 1024
):
    """Generate prompt for creating a full DMPO conversation at once."""
    return f"""# Full DMPO Conversation Generation

## TOPIC
{topic}

## TASK
Generate a complete {num_interactions}-interaction conversation with BOTH accepted and rejected responses for each step.

## REQUIREMENTS
1. {num_interactions} interconnected interactions testing memory
2. Each interaction has both accepted and rejected versions
3. Accepted: Excellent memory usage, strong reasoning, accurate answers
4. Rejected: Realistic errors (memory mistakes, reasoning flaws, wrong facts)
5. Progressive complexity - later interactions should require more memory synthesis

## ERROR TYPES FOR REJECTED RESPONSES (vary across interactions)
- Memory errors: Wrong facts from earlier, missing relevant context
- Reasoning errors: Logical fallacies, skipped steps, wrong conclusions
- Answer errors: Partially wrong, incomplete, or misleading responses
- Consistency errors: Contradict previous statements

## OUTPUT FORMAT
Output a Python list of dictionaries:
[
    {{
        "query": "Question 1...",
        "accepted": {{"think": "Good reasoning...", "answer": "Good answer..."}},
        "rejected": {{"think": "Flawed reasoning...", "answer": "Poor answer..."}}
    }},
    {{
        "query": "Question 2 (references Q1)...",
        "accepted": {{"think": "Correctly recalls Q1...", "answer": "Accurate answer..."}},
        "rejected": {{"think": "Misremembers Q1...", "answer": "Answer with error..."}}
    }},
    # ... {num_interactions} total interactions
]

Generate ONLY the Python list - no other text."""


# ============================================================================
# PROMPT TUPLES FOR DIFFERENT MODES
# ============================================================================

ALL_PROMPTS_REASONING_COMPLETION = (
    system_reasoning_completion_single,
    system_reasoning_completion_all,
    task_description_reasoning_completion_single,
    task_description_reasoning_completion_all,
)

ALL_PROMPTS_REASONING_GENERATION = (
    system_reasoning_generation_single,
    system_reasoning_generation_all,
    task_description_reasoning_generation_single,
    task_description_reasoning_generation_all,
)

ALL_PROMPTS_DMPO = (
    system_dmpo_generation_single,
    system_dmpo_generation_all,
    task_description_dmpo_single,
    task_description_dmpo_all,
)


def get_random_topics(n: int = 10, topics: list[str] = None) -> list[str]:
    """Get random topics for generation."""
    if topics is None:
        topics = TOPICS_HYBRID_REASONING
    return random.choices(topics, k=n)
