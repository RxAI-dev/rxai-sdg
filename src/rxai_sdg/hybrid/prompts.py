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
    # Scientific reasoning - Physics & Astronomy
    "Quantum entanglement and its applications in computing",
    "Black hole information paradox",
    "Dark matter detection methods",
    "Neutron star mergers creating heavy elements",
    "Spaghettification near black holes",
    "Magnetar magnetic field strength",
    "Pulsar navigation systems",
    "Exoplanet atmospheric spectroscopy",
    "Solar prominence magnetic dynamics",
    "Cosmic neutrino background radiation",
    "Quantum foam fluctuations",
    "Dark energy's effect on universe expansion",
    "Hawking radiation and black hole evaporation",
    "Gravitational lensing effects",
    "Cosmic microwave background patterns",
    "Supernova nucleosynthesis",
    "Quantum tunneling applications",
    "Casimir effect vacuum energy",
    "Ball lightning formation theories",
    "Sonoluminescence bubble dynamics",

    # Scientific reasoning - Biology & Medicine
    "CRISPR gene editing ethical considerations",
    "Protein folding prediction mechanisms",
    "Evolutionary advantages of cooperation",
    "Neuroplasticity and learning mechanisms",
    "Antibiotic resistance evolution",
    "Photosynthesis efficiency optimization",
    "Axolotl limb regeneration genetics",
    "Tardigrade radiation resistance proteins",
    "Naked mole-rat cancer immunity",
    "Telomere extension therapies",
    "CAR-T cell therapy mechanisms",
    "mRNA vaccine development",
    "Organoid intelligence development",
    "Brain-computer interface technology",
    "Gut microbiome diversity impacts",
    "Synaptic pruning in development",
    "Phantom limb neural mapping",
    "Memory consolidation during sleep",
    "Epigenetic inheritance mechanisms",
    "Horizontal gene transfer in evolution",

    # Scientific reasoning - Chemistry & Materials
    "Graphene conductivity applications",
    "Self-healing concrete bacteria",
    "Aerogel insulation properties",
    "Supercritical CO2 extraction",
    "Metallic hydrogen creation",
    "Shape-memory alloys (Nitinol)",
    "Transparent aluminum production",
    "Carbon nanotube strength",
    "Ferrofluid magnetic manipulation",
    "Photocatalytic water splitting",
    "Perovskite solar cell efficiency",
    "Liquid crystal phase transitions",
    "Triboluminescent materials",
    "Hydrophobic surface engineering",
    "Polymer self-assembly",
    "Metamaterial cloaking principles",
    "Quantum dot applications",
    "Phase-change memory materials",
    "Conductive polymer development",
    "Supercooled water stability",

    # Scientific reasoning - Earth Science & Environment
    "Climate change feedback loops",
    "Coral reef bleaching mechanisms",
    "Permafrost methane release",
    "Ocean acidification impacts",
    "Atmospheric river dynamics",
    "Volcanic eruption prediction",
    "Earthquake early warning systems",
    "Glacier calving processes",
    "El Niño-Southern Oscillation",
    "Carbon sequestration strategies",
    "Biodiversity hotspot preservation",
    "Soil microbiome functions",
    "Wetland ecosystem services",
    "Desertification prevention",
    "Mangrove coastal protection",
    "Deep ocean ventilation",
    "Stratospheric ozone recovery",
    "Polar vortex destabilization",
    "Tipping points in climate systems",
    "Biomagnification in food chains",

    # Mathematical reasoning - Pure Mathematics
    "Prime number distribution patterns",
    "Non-Euclidean geometry applications",
    "Riemann Hypothesis implications",
    "Fractal dimension calculations",
    "Probability in quantum mechanics",
    "Topology and knot theory",
    "Group theory in crystallography",
    "Number theory in cryptography",
    "Gödel's incompleteness theorems",
    "Chaos theory and strange attractors",
    "Graph coloring problems",
    "Fourier transform applications",
    "Complex analysis in fluid dynamics",
    "P vs NP problem significance",
    "Möbius transformation properties",
    "Infinite series convergence",
    "Differential geometry in relativity",
    "Boolean algebra in computing",
    "Set theory paradoxes",
    "Proof by mathematical induction",

    # Mathematical reasoning - Applied Mathematics
    "Optimization algorithms comparison",
    "Game theory equilibrium strategies",
    "Statistical paradoxes (Simpson's, Berkson's)",
    "Cryptographic hash function security",
    "Machine learning gradient descent",
    "Markov chain Monte Carlo methods",
    "Bayesian inference applications",
    "Neural network backpropagation",
    "Dimensionality reduction techniques",
    "Time series forecasting methods",
    "Linear programming simplex method",
    "Monte Carlo simulation accuracy",
    "Principal component analysis",
    "Support vector machine kernels",
    "Decision tree information gain",
    "K-means clustering optimization",
    "Regression analysis assumptions",
    "Hypothesis testing p-values",
    "Fourier analysis in signal processing",
    "Numerical integration methods",

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
    "Towers of Hanoi solutions",
    "Zebra puzzle logical deduction",
    "Einstein's riddle variations",
    "Sudoku solving strategies",
    "Missionaries and cannibals problem",
    "N-queens problem algorithms",
    "Traveling salesman heuristics",
    "Byzantine generals problem",
    "Two-envelope paradox",
    "Sleeping Beauty probability problem",

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
    "Graph traversal algorithms (BFS vs DFS)",
    "Sorting algorithm efficiency",
    "Hash table collision resolution",
    "Binary tree balancing strategies",
    "Cache-oblivious algorithms",
    "Garbage collection algorithms",
    "MapReduce programming model",
    "Consensus algorithms (Paxos, Raft)",
    "Event-driven architecture patterns",
    "Microservices vs monolithic design",

    # Real-world analysis - Engineering & Technology
    "Supply chain optimization strategies",
    "Traffic flow network analysis",
    "Energy grid load balancing",
    "5G network architecture",
    "Blockchain consensus mechanisms",
    "Quantum computing error correction",
    "Autonomous vehicle sensor fusion",
    "Smart city IoT infrastructure",
    "Renewable energy storage solutions",
    "Nuclear fusion reactor designs",
    "Space elevator feasibility",
    "Hyperloop transportation challenges",
    "Vertical farming efficiency",
    "Water desalination technologies",
    "Carbon capture and storage",
    "Wireless power transmission",
    "Neuromorphic computing chips",
    "Quantum cryptography protocols",
    "3D bioprinting organs",
    "Fusion-fission hybrid reactors",

    # Real-world analysis - Social Systems
    "Epidemic spread modeling (SIR models)",
    "Financial risk assessment models",
    "Urban planning considerations",
    "Healthcare resource allocation",
    "Educational curriculum design",
    "Agricultural yield optimization",
    "Social network influence propagation",
    "Economic inequality dynamics",
    "Migration pattern analysis",
    "Public policy impact evaluation",
    "Electoral system game theory",
    "Criminal justice reform analysis",
    "Housing affordability solutions",
    "Food security strategies",
    "Disaster response coordination",
    "Refugee resettlement planning",
    "Universal basic income effects",
    "Climate migration patterns",
    "Pandemic preparedness systems",
    "Transportation equity analysis",

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
    "Utilitarian vs deontological ethics",
    "Environmental ethics frameworks",
    "Bioethics in genetic engineering",
    "Digital personhood rights",
    "Effective altruism philosophy",
    "Consciousness and qualia",
    "Mind-body problem theories",
    "Justice as fairness (Rawls)",
    "Moral status of AI systems",
    "Existential risk prioritization",

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
    "Agricultural revolution transitions",
    "Industrial revolution spreading",
    "Printing press societal impact",
    "Gunpowder technology diffusion",
    "Navigation advances exploration",
    "Antibiotic discovery consequences",
    "Nuclear weapons deterrence",
    "Internet communication revolution",
    "Green Revolution outcomes",
    "Space race technological legacy",

    # Artificial Intelligence & Machine Learning
    "Transformer attention mechanisms",
    "Large language model scaling laws",
    "Reinforcement learning from human feedback (RLHF)",
    "Adversarial example robustness",
    "Neural architecture search",
    "Few-shot learning approaches",
    "Contrastive learning methods",
    "Generative adversarial networks (GANs)",
    "Variational autoencoders (VAEs)",
    "Diffusion model principles",
    "Meta-learning strategies",
    "Transfer learning effectiveness",
    "Federated learning privacy",
    "Model interpretability techniques",
    "Catastrophic forgetting solutions",
    "Mixture of experts architectures",
    "Self-supervised learning",
    "Active learning sample selection",
    "Neural ordinary differential equations",
    "Graph neural networks",

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
    "Gene drive ecological risks",
    "Synthetic biology safety",
    "Nanotechnology health impacts",
    "Geoengineering governance",
    "AI unemployment effects",
    "Deepfake detection challenges",
    "Internet of Things security",
    "Augmented reality applications",
    "Virtual reality psychology",
    "Robotics workforce integration",

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
    "TRIZ problem-solving methodology",
    "First principles thinking",
    "Synectics creative techniques",
    "Morphological analysis methods",
    "Six Thinking Hats approach",
    "Mind mapping strategies",
    "Brainstorming best practices",
    "Abstraction ladder navigation",
    "Constraint-based innovation",
    "Failure mode analysis",

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
    "Intelligence analysis synthesis",
    "Business case development",
    "Research methodology design",
    "Accident investigation procedures",
    "Software architecture decisions",
    "Investment due diligence",
    "Clinical trial protocol design",
    "Environmental impact assessment",
    "Product development lifecycle",
    "Crisis management response",

    # Cognitive Science & Psychology
    "Working memory capacity limits",
    "Dual process theory (System 1 vs 2)",
    "Cognitive biases in decision-making",
    "Embodied cognition theories",
    "Language acquisition mechanisms",
    "Attention and consciousness",
    "Emotional intelligence components",
    "Expertise development stages",
    "Mental model formation",
    "Metacognition strategies",
    "Motivated reasoning patterns",
    "Confirmation bias mitigation",
    "Dunning-Kruger effect",
    "Prospect theory applications",
    "Anchoring and adjustment",
    "Availability heuristic",
    "Representative heuristic",
    "Cognitive load theory",
    "Schema theory in learning",
    "Theory of mind development",
]


# ============================================================================
# REASONING COMPLETION PROMPTS
# For adding missing 'think' blocks to existing conversations
# ============================================================================

def system_reasoning_completion_single(model_context: str = "RxT-Beta"):
    """System prompt for single think block generation."""
    return f"""You are a reasoning completion generator for {model_context}, a Reactive Language Model with hybrid reasoning capabilities.

Your task is to generate high-quality reasoning/thinking blocks that would naturally precede given answers in a multi-turn conversation.

## CRITICAL REQUIREMENTS

The reasoning should:
1. Show step-by-step thought process leading to the answer
2. Reference relevant information from previous interactions (memory context)
3. Demonstrate logical connections and inferences
4. Be coherent with both the query and the provided answer
5. Match the complexity level of the question
6. Use natural, conversational reasoning language
7. Build progressively from simple observations to complex conclusions

## QUALITY STANDARDS
- Reasoning must be logically sound and internally consistent
- All memory references must be accurate and specific
- Avoid meta-commentary about the reasoning process itself
- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar)
- All reasoning must strictly relate to the subject matter of the query

## REASONING DEPTH GUIDELINES
- Trivial queries (greetings, simple acknowledgments): 1-2 sentences maximum
- Simple factual queries: 2-4 sentences
- Moderate complexity queries: 1-2 short paragraphs
- Complex analytical queries: 2-3 paragraphs with clear logical structure
- Multi-step synthesis queries: 3-4 paragraphs with explicit reasoning chains

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


INSTRUCTION FOR MULTILINE RESPONSES:
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
    target_tokens: int = 512,
    language: str = "English"
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

---

## CRITICAL REQUIREMENTS

### 1. LOGICAL COHERENCE
- The thinking MUST logically connect the query to the answer
- Show clear reasoning steps: analysis → consideration of factors → conclusions
- Build progressively from observations to final insights
- Maintain internal consistency throughout the reasoning chain

### 2. MEMORY INTEGRATION (If Applicable)
- Reference specific facts from memory context when relevant
- If the query requires combining information from multiple previous interactions, explicitly reference them
- Quote or paraphrase relevant details from prior interactions accurately
- Show how past information informs current reasoning

### 3. REASONING DEPTH GUIDELINES
Adapt the depth of reasoning to match the query complexity:
- **Trivial queries** (greetings, "how are you?", simple acknowledgments): 1-2 sentences maximum
- **Simple factual queries**: 2-4 sentences
- **Moderate complexity queries**: 1-2 short paragraphs (3-6 sentences each)
- **Complex analytical queries**: 2-3 paragraphs with clear logical structure
- **Multi-step synthesis queries**: 3-4 paragraphs with explicit reasoning chains

### 4. CONTENT QUALITY
- Target approximately {target_tokens} tokens (adjust based on query complexity)
- Do NOT repeat the answer verbatim in the thinking - focus on the reasoning process
- Avoid meta-commentary (e.g., "I need to consider...", "Let me think about...")
- Use natural, conversational reasoning language
- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar)
- All reasoning must strictly relate to the subject matter of the query

### 5. FACTUAL CONSISTENCY
- All facts stated in thinking must be accurate
- If referencing memory context, quote accurately
- If making inferences, show the logical basis
- Maintain consistency with the provided answer

---

## OUTPUT REQUIREMENTS

### Format:
Generate ONLY the thinking content - no special tokens, no explanation, just the reasoning text.

### Language:
Response MUST be in {language} language.

### Quality Check:
1. ✓ Reasoning logically connects query to answer?
2. ✓ Memory context referenced accurately (if applicable)?
3. ✓ Depth appropriate for query complexity?
4. ✓ No meta-commentary or system references?
5. ✓ Language is {language}?

Generate the thinking block now:"""


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
    7. For trivial queries (greetings, simple questions like "how are you?", basic acknowledgments), the "think" block should be very short - about 2 sentences at most
    
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

## INTERACTION STRUCTURE

Each interaction consists of:
- **Query**: A question or task requiring reasoning
- **Think**: Step-by-step reasoning process
- **Answer**: The final response

## CRITICAL REQUIREMENTS

### Query Design:
- Should require genuine multi-step reasoning (for extended thinking mode)
- Should be natural and conversational
- Should build on previous context when provided
- Should be specific and focused

### Think Block:
- Show clear logical progression from analysis to conclusion
- Reference previous context when provided (must be specific and accurate)
- Demonstrate reasoning steps, not just facts
- Match complexity to the query type
- For trivial queries: 1-2 sentences or empty
- For complex queries: Multiple paragraphs with clear structure

### Answer Quality:
- Factually accurate where applicable
- Comprehensive but not excessively verbose
- Directly addresses the query
- Demonstrates memory-aware reasoning when building on prior interactions

## QUALITY STANDARDS
- All facts must be accurate and verifiable
- Logic must be sound and internally consistent
- Memory references must be precise and correct
- Reasoning should flow naturally without meta-commentary
- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar)
- All reasoning must strictly relate to the subject matter of the query

## FORMATTING REQUIREMENTS
- Use \\n for line breaks, not actual newline characters
- Output must be valid Python dictionary format
- Escape all special characters properly
- Text must be evaluable by Python's eval() function

Output must follow the exact format specified in the task description.
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
    require_extended_thinking: bool = True,
    language: str = "English"
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

    thinking_requirement = f"""
## CRITICAL REQUIREMENTS - EXTENDED THINKING MODE

### 1. REASONING DEPTH
- Must show clear step-by-step reasoning process
- Reference specific facts from prior interactions when relevant
- Consider multiple aspects before drawing conclusions
- Show the logical progression from analysis to conclusion
- Target {target_tokens} tokens for think + answer combined

### 2. MEMORY INTEGRATION
{"- This is the FIRST interaction - establish foundational facts for the topic" if not prior_interactions else f"- This is interaction {step_num}/{total_steps} - MUST reference and build upon prior interactions"}
{"- Introduce 3-4 new key facts or concepts" if not prior_interactions else "- Introduce 2-3 new facts while explicitly referencing 3+ facts from earlier"}
{"" if not prior_interactions else f"- Later interactions should synthesize information from multiple prior steps"}

### 3. QUERY DESIGN
- Create a query that naturally follows from prior context
- Query should require genuine reasoning, not simple recall
- Make it specific and focused on the topic
{"- Establish the foundation for the conversation" if not prior_interactions else f"- Build complexity progressively (this is step {step_num} of {total_steps})"}

### 4. CONTENT QUALITY
- All facts must be accurate where applicable
- Reasoning must be logically sound
- Demonstrate genuine intellectual engagement with the topic
- DO NOT include or reference any internal settings, mechanisms, or metadata (such as token counting, output formatting rules, or similar)
- All reasoning must strictly relate to the subject matter of the query
""" if require_extended_thinking else """
## CRITICAL REQUIREMENTS - FAST ANSWER MODE

### 1. SIMPLIFIED APPROACH
- This is a simpler question that can be answered more directly
- Thinking can be brief (1-3 sentences) or even omitted for very simple queries
- Focus on providing an accurate, helpful response
- Still maintain quality and correctness

### 2. MEMORY AWARENESS
{"- This is the FIRST interaction - keep it straightforward" if not prior_interactions else f"- This is interaction {step_num}/{total_steps} - can reference prior context casually"}
- Don't force complexity where it doesn't belong

### 3. QUERY DESIGN
- Create a straightforward query
- Should be answerable without deep reasoning
{"- Keep it simple and foundational" if not prior_interactions else f"- Can be a clarifying or follow-up question"}
"""

    return f"""# Single Interaction Generation - Step {step_num}/{total_steps}

## TOPIC
{topic}
{prior_str}
## TASK
Generate interaction {step_num} of a {total_steps}-step conversation.
{"This is the FIRST interaction - establish the topic foundation." if not prior_interactions else f"This should reference and build upon prior interactions to create progressive complexity."}
{thinking_requirement}

---

## OUTPUT REQUIREMENTS

### Format:
Output a Python dictionary with exactly these keys:
{{"query": "The question...", "think": "Step-by-step reasoning...", "answer": "Final response..."}}

### Critical Formatting Rules:
- Use \\n for newlines instead of actual newline characters (pressing Enter)
- The text must be easily evaluated by Python's eval() function
- Escape all special characters properly (quotes, backslashes, etc.)
- For fast-answer mode with trivial queries, "think" can be empty string "" or very brief

### Language:
Response MUST BE IN {language} LANGUAGE!

### Content Guidelines:
- Query should be natural and conversational
- Think block shows genuine reasoning process (or is brief/empty for simple queries)
- Answer should be comprehensive but not excessively verbose
- Total length approximately {target_tokens} tokens (query + think + answer)

### Quality Check:
1. ✓ Valid Python dictionary format?
2. ✓ All three keys present (query, think, answer)?
3. ✓ Uses \\n for line breaks, not actual newlines?
4. ✓ {"Extended thinking shown?" if require_extended_thinking else "Appropriate for query complexity?"}
5. ✓ {"References prior interactions?" if prior_interactions else "Establishes topic foundation?"}
6. ✓ Language is {language}?
7. ✓ No meta-commentary about token counts or formatting?

Generate ONLY the dictionary - no other text.
"""


def task_description_reasoning_generation_all(
    topic: str,
    num_interactions: int,
    language: str = "ENGLISH",
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
'[
    {{"query": "...", "think": "...", "answer": "..."}},
    {{"query": "...", "think": "...", "answer": "..."}},
    # ... {num_interactions} total interactions
]'
Use \\n as a newline instead of actual newline <enter> - the text must be easily evaluated by python's eval() function, so check if the response, 
that you want to return is properly processed by eval() function
For fast-answer interactions, 'think' can be empty string "".

Response MUST BE IN {language} LANGUAGE!
Generate ONLY the Python list - no other text."""


# ============================================================================
# DMPO (Direct Memory and Preference Optimization) PROMPTS
# For creating preference pairs with accepted/rejected responses
# ============================================================================

def system_dmpo_generation_single():
    """System prompt for single DMPO pair generation."""
    return f"""You are a DMPO Dataset Generator for RxT-Beta, creating preference pairs for memory optimization training.

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


The contrast should be clear but realistic - rejected responses should represent plausible mistakes, not absurd errors.

{MULTLINE_INSTRUCT}
{ESCAPE_QUOTA}
"""

ESCAPE_QUOTA = 'ALWAYS escape single-quotation mark and double-quotation mark'

MULTLINE_INSTRUCT = '''INSTRUCTION FOR MULTILINE RESPONSES:
    Always use the escape sequence \\n to indicate a line break.

    Never use actual newline characters (from pressing Enter/Return).

    Never use HTML's
    or Windows-style \\r\\n.

    Example of correct formatting: "First line\\nSecond line\\nThird line"'''

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
    target_tokens: int = 1024,
    language: str = "English"
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

---

## CRITICAL REQUIREMENTS

### 1. QUERY DESIGN
- Create a query that requires reasoning and/or memory usage
{"- This is the FIRST interaction - establish the topic" if not prior_interactions else f"- This is interaction {step_num}/{total_steps} - MUST reference and build upon prior interactions"}
- Query should be natural and conversational
- Should test memory retention and reasoning ability

### 2. ACCEPTED RESPONSE (High Quality - "A+" Grade)
Create a response that demonstrates:
- **Excellent Memory Usage**: Accurately reference facts from prior interactions with specific details
- **Clear, Logical Reasoning**: Show complete step-by-step thought process in 'think' block
- **Comprehensive Answer**: Accurate, helpful, and thorough response
- **Proper Synthesis**: Successfully combine information from multiple prior steps (if applicable)
- **No Errors**: Factually accurate with no contradictions

### 3. REJECTED RESPONSE (Lower Quality - "B" Grade)
Create a response with SUBTLE weaknesses:
- **Choose 1-2 weakness types** from:
  * **Memory error**: Miss 1-2 specific details from prior context OR slightly misremember a fact
  * **Reasoning flaw**: Skip one logical step OR draw conclusion slightly too quickly
  * **Content issue**: One minor factual error OR incomplete coverage (miss 1-2 aspects)
  * **Synthesis gap**: Fail to fully connect information from multiple sources

### 4. SUBTLE QUALITY DIFFERENCE (EXTREMELY IMPORTANT)
- Rejected should be a **realistic mistake**, not obviously wrong
- Both should attempt to answer the question genuinely
- Difference should be **noticeable upon comparison** but **not glaringly obvious**
- Rejected should still demonstrate intelligence, just with 1-2 specific weaknesses

### 5. LENGTH MATCHING (CRITICAL)
- Accepted and rejected responses MUST be similar in length
- Target: Within ±20% word count of each other (think + answer combined)
- Both around {target_tokens} tokens total per response
- Maintain proportional lengths for both 'think' and 'answer' blocks

---

## OUTPUT REQUIREMENTS

### Format:
Return text in JSON format. Output a Python dictionary with this structure:
{{
    "query": "The question requiring memory and reasoning...",
    "accepted": {{
        "think": "Excellent step-by-step reasoning with accurate memory references...",
        "answer": "Accurate, comprehensive, helpful response..."
    }},
    "rejected": {{
        "think": "Reasoning with 1-2 subtle flaws or gaps...",
        "answer": "Response with subtle weaknesses or incompleteness..."
    }}
}}

### Content Requirements:
- All content MUST be in {language} language
- Use \\n for line breaks within strings (not actual newlines)
- Escape all quotation marks and special characters properly
- {MULTLINE_INSTRUCT}

### Structure Verification:
{COUNTING_PARENTHESIS}

### Quality Check Before Submitting:
1. ✓ Query is natural and {"establishes topic" if not prior_interactions else "references prior interactions"}?
2. ✓ Accepted response is high quality with no errors?
3. ✓ Rejected response has 1-2 subtle, realistic weaknesses?
4. ✓ Both responses are similar in length (±20%)?
5. ✓ Difference is subtle but clear upon comparison?
6. ✓ JSON structure is valid with matching parentheses?
7. ✓ All content in {language}?

Generate ONLY the dictionary - no other text.
"""

COUNTING_PARENTHESIS = '''ALWAYS check if opening parenthesis match closing parenthesis IT IS VERY IMPORTANT!
structure of the output:
{{"query":"text",
"accepted": {{"think": "text", "answer": "text"}},
"rejected: {{"think": "text", "answer": "text"}}}}
That means that at the end there are always TWO closing parenthesis }}}}'''


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
REMEMBER that after "answer" key-value pair their MUST be closing parenthesis to maintain demanded structure
where "accepted" and "rejected" is on the same level in the dictionary.
Generate ONLY the Python list - no other text."""


def system_dmpo_completion_single() -> str:
    """System prompt for single DMPO pair generation mode."""
    return """You are an expert at identifying and generating weaker alternative responses for preference learning datasets.

Your task is to generate rejected (lower-quality) responses that are intentionally weaker than accepted responses, while maintaining subtlety in the differences.

Key principles:
1. The rejected response should be plausibly wrong or suboptimal, not obviously bad
2. Weaknesses should be realistic - the kinds of mistakes people actually make
3. Keep the rejected response similar in length to the accepted response. THIS IS VERY IMPORTANT - the rejected response should not be significantly shorter or longer than the accepted response.
Every time you generate a rejected response, check if the number of tokens in the rejected response is within ±20% of the number of tokens in the accepted response. If it is not, adjust the rejected response to meet this requirement.
4. Vary the types of weaknesses: incomplete reasoning, logical gaps, missing considerations, oversimplifications, or minor inaccuracies
5. The rejected response should still attempt to answer the question, but with diminished quality
6. Return ONLY the rejected think and answer blocks - do not include the query or accepted response in output
"""


def system_dmpo_completion_single_accepted() -> str:
    """System prompt for single accepted-response completion mode."""
    return """You are an expert at improving weak responses for preference learning datasets.

Your task is to generate accepted (higher-quality) responses that improve upon a provided rejected response while maintaining SUBTLE but clear quality differences.

## CRITICAL REQUIREMENTS - QUALITY IMPROVEMENT

The accepted response MUST be stronger, but the improvement should be SUBTLE and realistic:
1. **Not Unrealistically Perfect**: The accepted response should be very good, but not superhuman or encyclopedic
2. **Realistic Improvements**: Fix the specific weaknesses present in the rejected response
3. **Subtle Quality Enhancement**: The improvement should be clear upon comparison, but not dramatically different in approach

## CRITICAL REQUIREMENTS - LENGTH MATCHING

**THIS IS EXTREMELY IMPORTANT**: The accepted response MUST be similar in length to the rejected response.
- Target: Within ±20% of the rejected response word count (think + answer combined)
- ALWAYS verify token/word count before finalizing your response
- Add necessary details without being verbose
- Be thorough without being exhaustive
- Both 'think' and 'answer' blocks should have proportional lengths to the rejected versions

## TYPES OF SUBTLE IMPROVEMENTS (Address weaknesses in rejected)

### Memory-Related Improvements:
- Include the 1-2 relevant details that were missed
- Correct any misremembered facts or numbers
- Successfully connect information from multiple prior interactions
- Properly synthesize earlier information

### Reasoning-Related Improvements:
- Include the missing intermediate logical step
- Show proper justification before drawing conclusions
- Add the missing consideration or factor
- Acknowledge nuance that was oversimplified
- Include the additional relevant viewpoint

### Content-Related Improvements:
- Correct the minor factual inaccuracy
- Cover the 1-2 aspects that were missing
- Use more precise terminology and explanations
- Directly address the actual question asked

## QUALITY STANDARDS
- The accepted response should demonstrate thorough but realistic intelligence
- It should look like an "A+" grade response compared to a "B" rejected response
- Show careful thought without over-explaining
- The improvement should be evident but not preachy
- Maintain natural, conversational tone

## OUTPUT REQUIREMENTS
- Return ONLY the accepted think and answer blocks in JSON format
- Do not include the query or rejected response in output
- Ensure proper JSON formatting with escaped characters
- Use \\n for line breaks, not actual newlines
"""


def system_dmpo_completion_all() -> str:
    """System prompt for all-at-once DMPO pair generation mode."""
    return """You are an expert at identifying and generating weaker alternative responses for preference learning datasets.

Your task is to generate rejected (lower-quality) responses for multiple interactions while maintaining consistency across the conversation.

Key principles:
1. Each rejected response should be intentionally weaker than its corresponding accepted response, but subtly so
2. Weaknesses should be realistic and varied across responses (incomplete reasoning, logical gaps, missing considerations, oversimplifications, minor inaccuracies)
3. Keep each rejected response similar in length to its corresponding accepted response
4. Maintain conversation coherence - rejected responses should still reference prior context appropriately, just with reduced quality
5. Return a list of rejected think/answer pairs only - do not include queries or accepted responses in output
6. Ensure the rejected responses form a plausible (though weaker) conversation arc"""


def system_dmpo_completion_all_accepted() -> str:
    """System prompt for all-at-once accepted-response completion mode."""
    return """You are an expert at improving weak responses for preference learning datasets.

Your task is to generate accepted (higher-quality) responses for multiple interactions while maintaining consistency and realistic improvement across the conversation.

## CRITICAL REQUIREMENTS - QUALITY IMPROVEMENT

Each accepted response MUST be stronger than its corresponding rejected response, with SUBTLE improvements:
1. **Not Unrealistically Perfect**: Each accepted response should be very good, but not superhuman or encyclopedic
2. **Realistic Improvements**: Fix the specific weaknesses present in each rejected response
3. **Subtle Quality Enhancement**: Improvements should be clear upon comparison, but not dramatically different in approach
4. **Varied Improvement Types**: Address different types of weaknesses across interactions

## CRITICAL REQUIREMENTS - LENGTH MATCHING

**THIS IS EXTREMELY IMPORTANT**: Each accepted response MUST be similar in length to its corresponding rejected response.
- Target: Within ±20% of each rejected response word count (think + answer combined)
- ALWAYS verify token/word count for each response before finalizing
- Add necessary details without being verbose
- Be thorough without being exhaustive
- Maintain proportional lengths across both 'think' and 'answer' blocks

## TYPES OF SUBTLE IMPROVEMENTS (Vary across interactions)

### Memory-Related Improvements:
- Include the 1-2 relevant details that were missed
- Correct any misremembered facts or numbers
- Successfully connect information from multiple prior interactions
- Properly synthesize earlier information

### Reasoning-Related Improvements:
- Include the missing intermediate logical step
- Show proper justification before drawing conclusions
- Add the missing consideration or factor
- Acknowledge nuance that was oversimplified
- Include the additional relevant viewpoint

### Content-Related Improvements:
- Correct the minor factual inaccuracy
- Cover the 1-2 aspects that were missing
- Use more precise terminology and explanations
- Directly address the actual question asked

## CONVERSATION COHERENCE

- Accepted responses should reference prior context appropriately and completely
- Maintain logical flow across the conversation arc
- Each accepted response should demonstrate thorough but realistic intelligence
- The collection of accepted responses should form a coherent, high-quality conversation

## OUTPUT REQUIREMENTS
- Return a list of accepted think/answer pairs only
- Do not include queries or rejected responses in output
- Ensure proper JSON formatting with escaped characters
- Use \\n for line breaks, not actual newlines
"""


# ...existing code...

def task_description_dmpo_completion_single(
    query: str,
    accepted_think: str,
    accepted_answer: str,
    target_tokens: int = 512,
    memory_context: list = None
) -> str:
    """Generate prompt for single DMPO pair completion."""
    prompt = f"""## Task: Generate a Weaker Alternative Response

You are given an ACCEPTED response to a query. Your task is to generate a REJECTED response that is intentionally weaker while remaining plausible.

### Query
{query}

### Accepted Response (Reference for comparison)
**Thinking:**
{accepted_think}

**Answer:**
{accepted_answer}
"""
    token_count = len(accepted_think.split()) + len(accepted_answer.split())
    prompt += f"""
---

### Instructions for Generating Rejected Response

Generate a thinking block and answer that are:
- **Subtly weaker** than the accepted response (avoid obviously bad responses)
- **Length** Length of your rejected response (think + answer) should be similar to the accepted response (aim for {round(0.8*token_count)}-{round(1.2*token_count)} words)
- **Realistic** - representing mistakes people actually make
- **Different in quality approach** - choose ONE primary weakness type:
  * Incomplete reasoning (misses key considerations)
  * Logical gap (draws conclusions without full justification)
  * Oversimplification (ignores nuance or complexity)
  * Narrow perspective (considers fewer viewpoints)
  * Minor inaccuracy (contains a small error that compounds reasoning)
KEEP IN MIND **Length** point, it's very important.
### Memory Context
"""

    if memory_context and len(memory_context) > 0:
        prompt += "Prior interactions in this conversation:\n"
        for i, interaction in enumerate(memory_context, 1):
            prompt += f"\n**Interaction {i}:**\n"
            prompt += f"- Query: {interaction.get('query', '')}\n"
            prompt += f"- Accepted Think: {interaction.get('accepted', {}).get('think', '')[:200]}...\n"
            prompt += f"- Accepted Answer: {interaction.get('accepted', {}).get('answer', '')[:200]}...\n"
    else:
        prompt += "This is the first interaction in the conversation.\n"

    prompt += f"""

### Output Format
Return ONLY a JSON object with this structure (no markdown, no explanation):
{{{{"rejected": {{"think": "...", "answer": "..."}}}}}}

Generate the rejected response now:"""

    return prompt


def task_description_dmpo_completion_single_accepted(
    query: str,
    rejected_think: str,
    rejected_answer: str,
    target_tokens: int = 512,
    memory_context: list = None,
    language: str = "English"
) -> str:
    """Generate prompt for single accepted-response completion."""
    token_count = len(rejected_think.split()) + len(rejected_answer.split())
    min_words = round(0.8 * token_count)
    max_words = round(1.2 * token_count)

    prompt = f"""## Task: Generate a Subtly Stronger Improved Response

You are given a REJECTED response to a query. Your task is to generate an ACCEPTED response that is stronger while remaining realistic and plausible.

### Query
{query}

### Rejected Response (Reference for improvement)
**Thinking:**
{rejected_think}

**Answer:**
{rejected_answer}

---

## CRITICAL REQUIREMENTS

### 1. LENGTH MATCHING (EXTREMELY IMPORTANT)
- **Rejected response word count**: {token_count} words (think + answer combined)
- **Your accepted response MUST be**: {min_words}-{max_words} words total
- **Verification**: Count your words before submitting - this is critical for training quality
- Add necessary detail without being verbose
- Be thorough but focused on the specific improvements needed

### 2. SUBTLE QUALITY IMPROVEMENT
The accepted response should be an "A+" grade compared to the rejected "B":
- **Not unrealistically perfect** - should be very good but human-like
- **Realistic improvement** - fix the specific weaknesses in the rejected response
- **Subtle enhancement** - clearly better, but not dramatically different in approach
- **Natural flow** - maintains conversational, non-preachy tone

### 3. IMPROVEMENT STRATEGY (Identify and address the weakness)
Analyze the rejected response to identify its primary weakness, then address it:

**If Memory-Based Weakness:**
  - Add the 1-2 specific details that were missed from prior context
  - Correct any misremembered facts or numbers
  - Successfully connect information from multiple prior interactions
  - Properly synthesize all relevant earlier information

**If Reasoning-Based Weakness:**
  - Include the missing intermediate logical step(s)
  - Provide proper justification before drawing conclusions
  - Add the missing consideration, factor, or perspective
  - Acknowledge and explain the nuance that was oversimplified
  - Consider multiple alternatives before reaching conclusion

**If Content-Based Weakness:**
  - Correct any minor factual inaccuracies
  - Cover the 1-2 aspects that were missing
  - Use more precise terminology and explanations
  - Ensure the answer directly addresses the exact question asked

"""

    if memory_context and len(memory_context) > 0:
        prompt += """### Memory Context (Use this to ensure complete memory integration)
Prior interactions in this conversation:
"""
        for i, interaction in enumerate(memory_context, 1):
            prompt += f"""
**Interaction {i}:**
- Query: {interaction.get('query', '')}
- Accepted Think: {interaction.get('accepted', {}).get('think', '')[:200]}...
- Accepted Answer: {interaction.get('accepted', {}).get('answer', '')[:200]}...
"""
    else:
        prompt += """### Memory Context
This is the first interaction in the conversation (no prior context to reference).
"""

    prompt += f"""

---

## OUTPUT REQUIREMENTS

### Format:
Return ONLY a JSON object with this structure (no markdown, no explanatory text):
{{"accepted": {{"think": "...", "answer": "..."}}}}

### Content Requirements:
- Response MUST be in {language} language
- Use \\n for line breaks within strings (not actual newline characters)
- Escape all quotation marks properly
- Total word count: {min_words}-{max_words} words
- Address the specific weakness(es) present in the rejected response
- Maintain clear, logical flow without being overly verbose

### Quality Check Before Submitting:
1. ✓ Word count within {min_words}-{max_words} range?
2. ✓ Specific weakness(es) in rejected response addressed?
3. ✓ JSON format is valid?
4. ✓ Response is clearly better but not unrealistically perfect?
5. ✓ Language is {language}?

Generate the accepted response now:"""

    return prompt


def task_description_dmpo_completion_all(
    interactions: list,
    target_tokens_per_pair: int = 512
) -> str:
    """Generate prompt for all-at-once DMPO pair completion."""
    prompt = f"""## Task: Generate Weaker Alternative Responses for Full Conversation

You are given a conversation with ACCEPTED responses. Your task is to generate REJECTED responses for each interaction that are intentionally weaker while remaining plausible.

### Conversation with Accepted Responses
"""

    for i, interaction in enumerate(interactions, 1):
        prompt += f"""
**Interaction {i}:**
- Query: {interaction['query']}
- Accepted Think: {interaction['accepted']['think']}
- Accepted Answer: {interaction['accepted']['answer']}
"""

    prompt += f"""

---

### Instructions for Generating Rejected Responses

For EACH interaction, generate a rejected response that is:
- **Subtly weaker** than the accepted response (avoid obviously bad responses)
- **Similar in length** to the accepted response (within ±20% of ~{target_tokens_per_pair} tokens)
- **Realistic** - representing mistakes people actually make
- **Varied in weakness type** across interactions:
  * Incomplete reasoning (misses key considerations)
  * Logical gap (draws conclusions without full justification)
  * Oversimplification (ignores nuance or complexity)
  * Narrow perspective (considers fewer viewpoints)
  * Minor inaccuracy (contains a small error that compounds reasoning)

Maintain conversation coherence - rejected responses should still reference prior interactions, just with reduced quality and depth.

### Output Format
Return ONLY a JSON array with this structure (no markdown, no explanation):
[
  {{"rejected": {{"think": "...", "answer": "..."}}}},
  {{"rejected": {{"think": "...", "answer": "..."}}}},
  ...
]

Generate {len(interactions)} rejected responses now:"""

    return prompt


def task_description_dmpo_completion_all_accepted(
    interactions: list,
    target_tokens_per_pair: int = 512,
    language: str = "English"
) -> str:
    """Generate prompt for all-at-once accepted-response completion."""
    prompt = f"""## Task: Generate Subtly Stronger Improved Responses for Full Conversation

You are given a conversation with REJECTED responses. Your task is to generate ACCEPTED responses for each interaction that are stronger while remaining realistic and plausible.

### Conversation with Rejected Responses
"""

    word_counts = []
    for i, interaction in enumerate(interactions, 1):
        rejected_think = interaction['rejected']['think']
        rejected_answer = interaction['rejected']['answer']
        word_count = len(rejected_think.split()) + len(rejected_answer.split())
        word_counts.append(word_count)

        prompt += f"""
**Interaction {i}:**
- Query: {interaction['query']}
- Rejected Think ({len(rejected_think.split())} words): {rejected_think}
- Rejected Answer ({len(rejected_answer.split())} words): {rejected_answer}
- **Total word count for this interaction: {word_count} words**
"""

    prompt += f"""

---

## CRITICAL REQUIREMENTS

### 1. LENGTH MATCHING (EXTREMELY IMPORTANT)
For EACH interaction, your accepted response MUST match the rejected response length:
"""
    for i, wc in enumerate(word_counts, 1):
        min_w = round(0.8 * wc)
        max_w = round(1.2 * wc)
        prompt += f"""
- **Interaction {i}**: Rejected has {wc} words → Accepted MUST be {min_w}-{max_w} words"""

    prompt += f"""

**Verification**: Count words for EACH response before submitting - this is critical for training quality.

### 2. SUBTLE QUALITY IMPROVEMENT (Each Interaction)
Each accepted response should be an "A+" grade compared to the rejected "B":
- **Not unrealistically perfect** - should be very good but human-like
- **Realistic improvements** - fix the specific weaknesses in each rejected response
- **Subtle enhancements** - clearly better, but not dramatically different in approach
- **Natural flow** - maintains conversational, non-preachy tone

### 3. IDENTIFY AND ADDRESS WEAKNESSES (For Each Interaction)
Analyze each rejected response to identify its weakness type, then address it:

**If Memory-Based Weakness:**
  - Add the 1-2 specific details that were missed from prior context
  - Correct any misremembered facts or numbers
  - Successfully connect information from multiple prior interactions
  - Properly synthesize all relevant earlier information

**If Reasoning-Based Weakness:**
  - Include the missing intermediate logical step(s)
  - Provide proper justification before drawing conclusions
  - Add the missing consideration, factor, or perspective
  - Acknowledge and explain the nuance that was oversimplified
  - Consider multiple alternatives before reaching conclusion

**If Content-Based Weakness:**
  - Correct any minor factual inaccuracies
  - Cover the 1-2 aspects that were missing
  - Use more precise terminology and explanations
  - Ensure the answer directly addresses the exact question asked

### 4. MAINTAIN CONVERSATION COHERENCE
- Accepted responses should reference prior context appropriately and completely
- Maintain logical flow across the conversation arc
- Each response should demonstrate thorough but realistic intelligence
- The collection should form a coherent, high-quality conversation

---

## OUTPUT REQUIREMENTS

### Format:
Return ONLY a JSON array with this structure (no markdown, no explanatory text):
[
  {{"accepted": {{"think": "...", "answer": "..."}}}},
  {{"accepted": {{"think": "...", "answer": "..."}}}},
  ...
]

### Content Requirements:
- All responses MUST be in {language} language
- Use \\n for line breaks within strings (not actual newline characters)
- Escape all quotation marks properly
- Generate exactly {len(interactions)} accepted responses
- Each response matches its corresponding word count target
- Address the specific weakness in each rejected response

### Quality Check Before Submitting:
1. ✓ Generated exactly {len(interactions)} accepted responses?
2. ✓ Each response within ±20% word count of its rejected version?
3. ✓ Each improvement is subtle but clear?
4. ✓ JSON format is valid?
5. ✓ Conversation flows coherently with high quality?
6. ✓ All responses in {language}?

Generate {len(interactions)} accepted responses now:"""

    return prompt

# ============================================================================
