"""
Few-shot examples for RxT-Beta Hybrid Reasoning Dataset Generators

These examples demonstrate:
1. Reasoning completion - Adding think blocks to existing conversations
2. Reasoning generation - Creating new conversations with hybrid reasoning
3. DMPO pairs - Accepted vs rejected responses for preference learning

Each example shows the expected quality and format for generated data.
Examples are designed for up to 8192 tokens per interaction (RxT-Beta scale).
"""

# ============================================================================
# REASONING COMPLETION EXAMPLES
# Show how to generate missing 'think' blocks for existing Q&A pairs
# ============================================================================

EXAMPLES_REASONING_COMPLETION_SINGLE: dict[str, str] = {
    "basic": '''
Given query: "What causes the seasons to change on Earth?"
Given answer: "The seasons change primarily due to Earth's axial tilt of approximately 23.5 degrees relative to its orbital plane around the Sun. When the Northern Hemisphere is tilted toward the Sun, it experiences summer because sunlight hits at a more direct angle, spreading less and delivering more energy per unit area. Simultaneously, the Southern Hemisphere experiences winter. The opposite occurs six months later when Earth has moved to the other side of its orbit."

Generated thinking:
The question asks about seasonal changes, which is fundamentally a question about Earth's orbital mechanics and geometry. Let me work through the key factors:

First, I should consider what doesn't cause seasons - many people incorrectly think it's about distance from the Sun, but Earth's orbit is nearly circular, and we're actually slightly closer to the Sun during Northern Hemisphere winter.

The real cause is Earth's axial tilt. The axis is tilted at about 23.5 degrees from perpendicular to our orbital plane. This tilt stays pointing in the same direction (toward Polaris) throughout the year.

Now, how does this create seasons? When one hemisphere is tilted toward the Sun:
1. Sunlight hits at a more direct angle (closer to perpendicular)
2. The same amount of solar energy is spread over a smaller area
3. Days are longer, providing more total heating time
4. This combination creates summer conditions

The opposite hemisphere simultaneously experiences:
1. Sunlight hitting at an oblique angle
2. Energy spread over larger area (less intensity)
3. Shorter days
4. Winter conditions

This explains why seasons are opposite in the Northern and Southern Hemispheres, and why the transition points (equinoxes) occur when neither hemisphere is preferentially tilted toward the Sun.
''',

    "memory_reference": '''
Memory Context:
- Interaction 1: Q: "How does photosynthesis work?" A: "Photosynthesis converts CO2 and water into glucose and oxygen using sunlight. The light reactions in chloroplasts capture solar energy and split water molecules, while the Calvin cycle uses this energy to fix carbon dioxide into sugar molecules. The process produces approximately 6 molecules of oxygen for every glucose molecule synthesized."

Given query: "Why are plants green rather than black, which would absorb more light?"
Given answer: "While black pigments would indeed absorb more total light, plants evolved green chlorophyll that appears to be highly optimized for Earth's conditions. The green color results from chlorophyll absorbing red and blue light while reflecting green. This may have evolved because early photosynthetic organisms that used different wavelengths already occupied the most light-rich environments, or because the energy in green light is actually less useful for the specific chemistry of photosynthesis. Additionally, absorbing all wavelengths could cause overheating damage and produce harmful reactive oxygen species."

Generated thinking:
This is an interesting follow-up to our earlier discussion of photosynthesis. The user is asking about an apparent inefficiency in plant evolution - why not absorb all light?

From our previous interaction, I recall that photosynthesis uses light reactions in chloroplasts to capture solar energy. The key insight is that "capturing solar energy" doesn't necessarily mean "absorbing all available light."

Let me consider several hypotheses for why plants are green:

1. Historical/evolutionary constraints: The earliest photosynthetic bacteria may have used purple pigments (bacteriorhodopsin), occupying the green-absorbing niche. Later organisms evolved to use the remaining wavelengths - red and blue.

2. Energy optimization: Different wavelengths carry different energy levels. Blue light is high-energy and can damage cellular structures. Red light has the right energy for the specific electron transitions in chlorophyll chemistry. Green light's intermediate energy may be less optimal.

3. Damage prevention: If plants absorbed all light, they'd need to convert or dissipate all that energy. Excess energy creates reactive oxygen species that damage cells. Reflecting some light (green) may be a protective mechanism.

4. Thermal regulation: Black absorption would cause significant heating, potentially denaturing proteins and disrupting the Calvin cycle we discussed earlier.

The answer should acknowledge that evolution doesn't always produce "optimal" solutions by human engineering standards, but rather solutions that are good enough given constraints and historical contingencies.
''',

    "multi_step_reference": '''
Memory Context:
- Interaction 1: Q: "What are the main causes of the French Revolution?" A: "The French Revolution (1789-1799) had multiple interconnected causes: financial crisis from expensive wars including support for American independence, severe inequality in the Three Estates system where clergy and nobility paid few taxes while commoners bore the burden, Enlightenment ideas spreading concepts of individual rights and social contract theory, and food shortages from poor harvests that drove up bread prices. The king's inability to reform and his calling of the Estates-General after 175 years created the catalyst for revolutionary action."

- Interaction 2: Q: "How did Enlightenment philosophy specifically influence revolutionary ideas?" A: "Enlightenment thinkers provided the intellectual framework for revolution. Rousseau's social contract theory argued that legitimate government requires the consent of the governed. Montesquieu advocated separation of powers to prevent tyranny. Voltaire championed religious tolerance and civil liberties. These ideas spread through salons and pamphlets, convincing many that the existing order was not divinely ordained but could be changed. The American Revolution's success demonstrated these theories could work in practice."

Given query: "Why did the revolution become increasingly radical after 1792?"
Given answer: "The revolution radicalized due to converging crises: foreign invasion by Austria and Prussia threatened to restore the monarchy, internal counterrevolution in the Vendée required harsh suppression, and economic conditions worsened. The sans-culottes (urban workers) pressured for more extreme measures. Political factions competed to prove revolutionary credentials, leading to the Terror under the Committee of Public Safety. Each crisis justified more radical responses, creating a cycle where moderation became suspect and potential grounds for execution. The guillotine claimed both supposed enemies and eventually the revolutionaries themselves."

Generated thinking:
This question builds directly on our previous discussions of the Revolution's causes and Enlightenment influence. I need to trace how the initially hopeful revolution of 1789 became the Terror of 1793-94.

From Interaction 1, I recall the key initial causes: financial crisis, inequality of the Three Estates, and Enlightenment ideas combined with practical triggers like food shortages. The revolution initially seemed to fulfill Enlightenment promises - a constitutional monarchy, Declaration of Rights of Man.

From Interaction 2, I noted how Enlightenment ideas about consent of governed (Rousseau) and separation of powers (Montesquieu) had inspired the revolutionaries. The American success proved these theories workable.

So why did things radicalize after 1792? Let me trace the escalation:

1. External threats: Austria and Prussia's declaration aimed to restore Louis XVI created existential threat. War hysteria breeds radicalism - you can't be moderate when invasion looms.

2. The King's betrayal: Louis's failed flight to Varennes (1791) and his correspondence with foreign powers proved he wasn't a good-faith constitutional monarch. This undermined the moderate position.

3. Economic conditions: The war economy, combined with continued bread shortages mentioned in our first discussion, kept the sans-culottes angry and mobilized.

4. Political competition: Without established democratic norms, factions accused each other of being insufficiently revolutionary. Girondins fell to Jacobins, who eventually fell to their own purity tests.

5. The Enlightenment ideas themselves: Rousseau's concept of "general will" could justify forcing people to be free. Virtue became the standard, and virtue is hard to measure - leading to paranoia.

The Terror emerged from this convergence: real threats, political competition, popular pressure, and ideological justification all pushing toward extremism.
'''
}

EXAMPLES_REASONING_COMPLETION_ALL: str = '''
Input conversation:
[
    {"query": "What is machine learning?", "answer": "Machine learning is a subset of artificial intelligence where systems learn patterns from data rather than being explicitly programmed. The three main types are supervised learning (labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error with rewards). Modern ML powers everything from recommendation systems to language models."},
    {"query": "How does a neural network actually learn?", "answer": "Neural networks learn through backpropagation and gradient descent. During forward pass, input data flows through layers of interconnected nodes, each applying weights and activation functions. The output is compared to the target, producing an error. Backpropagation calculates how much each weight contributed to this error by computing gradients layer by layer. Gradient descent then adjusts weights in the direction that reduces error. Over many iterations with different training examples, the network develops weights that generalize to new data."},
    {"query": "Why do deep networks work better than shallow ones for complex tasks?", "answer": "Deep networks excel at complex tasks because each layer can learn increasingly abstract representations. Early layers detect simple patterns (edges in images, phonemes in speech), while deeper layers combine these into complex concepts (faces, words). This hierarchical feature learning mirrors how the visual cortex processes information. Mathematically, deep networks can represent certain functions exponentially more efficiently than shallow ones - a shallow network would need exponentially many nodes to represent what a deep network achieves with fewer total parameters. Additionally, skip connections and batch normalization solved earlier training difficulties."}
]

Generated thinking blocks:
[
    "The user is asking a foundational question about machine learning. I should provide a clear, structured overview.\n\nLet me think about what defines ML and distinguishes it from traditional programming:\n\n1. Traditional programming: Human writes explicit rules\n2. Machine learning: System discovers rules from data\n\nThis is the fundamental paradigm shift. Now, what are the main approaches?\n\n- Supervised: We have input-output pairs, system learns the mapping\n- Unsupervised: No labels, system finds structure in data\n- Reinforcement: Learning through interaction and feedback\n\nI should also mention practical applications to ground this abstract concept in reality. Recommendation systems (Netflix, Spotify), language models (like me!), computer vision - these make ML tangible.",

    "The user wants to understand the learning mechanism itself - how does a pile of numbers actually learn? This builds on our previous answer about what ML is.\n\nLet me trace through the learning process step by step:\n\n1. Forward pass: Data enters, gets transformed by each layer's weights and activations, produces output. At this point, the network is just computing.\n\n2. Loss calculation: Compare output to target. This quantifies 'how wrong' the network is.\n\n3. Backward pass (backpropagation): The key insight! Use calculus (chain rule) to determine how each weight contributed to the error. Weights that pushed toward wrong answer get negative gradients.\n\n4. Update step (gradient descent): Move each weight a small step in the direction that reduces error.\n\n5. Repeat: Many examples, many iterations, gradually improving.\n\nThe beautiful thing is that this simple process - calculate error, distribute blame, adjust - can learn incredibly complex functions. The 'learning' emerges from massive repetition of this cycle.",

    "Now the user is asking why depth matters - why do we stack many layers instead of making one layer very wide? This connects to our previous discussion of backpropagation through layers.\n\nLet me consider this from multiple angles:\n\n1. Hierarchical representations: Real-world concepts have hierarchy. A 'face' is made of eyes, nose, mouth. An 'eye' is made of curves and colors. Deep networks naturally learn this hierarchy - early layers get simple features, deeper layers compose them.\n\n2. Efficiency argument: There's a mathematical result that some functions require exponentially many nodes in a shallow network but polynomially many in a deep one. Deep networks are more parameter-efficient for certain function classes.\n\n3. Biological inspiration: The visual cortex has layers (V1, V2, V4, IT) that process increasingly abstract features. Deep networks mirror this organization.\n\n4. Historical context: Deep networks weren't always trainable - vanishing gradients killed learning in early layers. Skip connections (ResNet) and batch normalization solved this, unlocking depth.\n\nI should connect this back to our backpropagation discussion - the gradient flow through layers is what makes depth both powerful and tricky to train."
]
'''


# ============================================================================
# REASONING GENERATION EXAMPLES
# Show how to create new conversations with hybrid reasoning from scratch
# ============================================================================

EXAMPLES_REASONING_GENERATION_SINGLE: dict[int, str] = {
    1: '''
Topic: Quantum entanglement applications

Generated interaction 1 (first of 3-step conversation):
{
    "query": "What is quantum entanglement and why do physicists consider it so strange?",
    "think": "The user is asking about quantum entanglement - one of the most counterintuitive phenomena in physics. I should explain both what it is and why it's considered 'strange' compared to classical intuition.\n\nLet me structure my thinking:\n\n1. Basic definition: Entanglement is a quantum mechanical phenomenon where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently.\n\n2. What makes it strange:\n   - Measurement correlation: When you measure one particle, you instantly know the state of its partner\n   - Non-locality: This works regardless of distance - could be light-years apart\n   - Einstein's objection: He called it 'spooky action at a distance' because it seemed to violate locality\n   - Bell's theorem: Proved this isn't just hidden variables - the correlations are genuinely stronger than any classical mechanism could produce\n\n3. Important clarification: It doesn't transmit information faster than light - you still need classical communication to compare results\n\nI should present this in a way that captures the genuine strangeness while being accurate about what entanglement can and cannot do.",
    "answer": "Quantum entanglement occurs when two or more particles become correlated in a fundamental way that has no classical equivalent. When particles are entangled, measuring a property of one instantly determines the corresponding property of its partner, regardless of the distance separating them.\n\nPhysicists find this strange because it defies our everyday intuitions about locality - the idea that objects are only directly influenced by their immediate surroundings. Einstein famously called it 'spooky action at a distance' and believed it indicated quantum mechanics was incomplete.\n\nWhat makes entanglement genuinely mysterious, confirmed by experiments testing Bell's inequalities, is that the correlations between entangled particles are stronger than any possible classical explanation. It's not that the particles carried hidden predetermined values - they truly don't have definite states until measured, yet somehow 'coordinate' their random outcomes across any distance.\n\nImportantly, this doesn't allow faster-than-light communication. While the correlation is instantaneous, you can't control which outcome you get, and you need classical communication to verify the correlation occurred."
}
''',
    2: '''
Topic: Quantum entanglement applications
Prior interaction: (Question about what entanglement is and why it's strange)

Generated interaction 2 (second of 3-step conversation):
{
    "query": "Given how strange entanglement is, what practical applications are scientists actually pursuing?",
    "think": "Building on our previous discussion of entanglement's counterintuitive properties, the user now wants to know about practical applications. I should connect the strange features we discussed to why they're useful.\n\nFrom our previous exchange, I established:\n- Entanglement creates correlations stronger than classical mechanisms\n- Measurement of one particle instantly affects knowledge of the other\n- Einstein's concern about 'spooky action at a distance'\n\nNow, what applications leverage these properties?\n\n1. Quantum cryptography (QKD):\n   - Uses entanglement for key distribution\n   - Any eavesdropping disturbs the entanglement, revealing the spy\n   - Leverages the 'measurement affects state' property we discussed\n\n2. Quantum computing:\n   - Entanglement enables quantum parallelism\n   - Entangled qubits can represent exponentially many states\n   - Algorithms like Shor's (factoring) and Grover's (search) use entanglement\n\n3. Quantum teleportation:\n   - Transfers quantum state using entanglement + classical communication\n   - Doesn't violate no-FTL (requires classical channel)\n   - Essential for quantum networks\n\n4. Quantum sensing:\n   - Entangled particles can measure with precision beyond classical limits\n   - Applications in GPS, medical imaging, gravitational wave detection\n\nI should emphasize that these aren't science fiction - some are already deployed (QKD) while others are in active development.",
    "answer": "The strange properties of entanglement that we discussed actually make it incredibly useful for several practical applications:\n\n**Quantum Key Distribution (QKD)**: Perhaps the most mature application. Because any attempt to intercept entangled particles disturbs their quantum state, entanglement enables provably secure communication. If an eavesdropper tries to measure the particles, the disturbance is detectable. Several companies already sell QKD systems, and China has demonstrated satellite-based quantum key distribution over 1,200 km.\n\n**Quantum Computing**: Entanglement is essential for quantum computational advantage. When qubits are entangled, they can exist in superpositions that represent exponentially many states simultaneously. Algorithms like Shor's algorithm (which could break RSA encryption) and Grover's search algorithm fundamentally require entanglement to achieve their speedups.\n\n**Quantum Teleportation**: This uses entanglement plus classical communication to transfer a quantum state perfectly from one location to another - essential for future quantum networks. Notably, this respects the speed of light limit we mentioned earlier, as the classical communication is still needed.\n\n**Quantum Sensing**: Entangled particles can achieve measurement precision beyond classical limits (the standard quantum limit). Applications include ultra-precise clocks, enhanced GPS systems, improved medical imaging, and gravitational wave detectors like LIGO.\n\nEach application cleverly exploits the very features that make entanglement seem strange - the correlations, the sensitivity to observation, and the quantum nature of information."
}
''',
    3: '''
Topic: Quantum entanglement applications
Prior interactions: (1. What entanglement is; 2. Practical applications)

Generated interaction 3 (final of 3-step conversation):
{
    "query": "You mentioned quantum key distribution is already deployed. How exactly does it work, and could quantum computers eventually break it too?",
    "think": "This is a great synthesis question that connects multiple threads from our conversation:\n1. From interaction 1: The strange properties of entanglement (measurement disturbs state)\n2. From interaction 2: QKD as a practical application; mention of Shor's algorithm breaking RSA\n\nThe user wants to understand:\na) How QKD actually works mechanically\nb) Whether quantum computers could break quantum cryptography (ironic if so!)\n\nLet me think through QKD protocols:\n\nBB84 protocol (original, uses single photons):\n- Alice sends photons in random bases\n- Bob measures in random bases\n- They compare bases publicly, keep matching ones\n- Any interception causes errors they can detect\n\nE91 protocol (entanglement-based):\n- Source creates entangled pairs\n- Alice and Bob each get one particle\n- They perform measurements in various bases\n- Bell inequality violations confirm no eavesdropper\n\nRegarding quantum computers breaking QKD:\n- Shor's algorithm breaks RSA (classical crypto) by factoring\n- QKD security is based on physics (uncertainty principle), not mathematical hardness\n- Quantum computers actually strengthen the need for QKD!\n- There are potential attacks (side channels, implementation flaws) but not fundamental breaks\n\nI should connect this back to our earlier discussion of why entanglement measurement is inherently disturbing - that's the core security principle.",
    "answer": "Let me explain how QKD works, connecting back to the entanglement properties we've discussed.\n\n**How QKD Works (E91 Protocol)**:\n1. A source generates entangled photon pairs and sends one photon to Alice, one to Bob\n2. Each independently measures their photon's polarization in randomly chosen bases\n3. They publicly compare which bases they used (not results) and keep only matching measurements\n4. They can test Bell inequalities on a subset - violations confirm genuine entanglement and no eavesdropper\n5. Any eavesdropper (Eve) trying to intercept and measure would disturb the entanglement, reducing the Bell violation below the quantum threshold\n6. This disturbance manifests as errors in the remaining shared bits, revealing Eve's presence\n\nThe security fundamentally relies on what we discussed earlier - measurement disturbs quantum states, and entanglement correlations are stronger than anything achievable through hidden classical mechanisms.\n\n**Could Quantum Computers Break QKD?**\n\nInterestingly, no - and this is important! The threat model is different:\n\n- **Classical cryptography (RSA, ECC)**: Security based on computational hardness (factoring, discrete log). Shor's algorithm running on a large quantum computer would break these.\n\n- **QKD**: Security based on laws of physics (uncertainty principle, no-cloning theorem). Even a perfect quantum computer can't violate these physical principles.\n\nIn fact, quantum computers make QKD more important, not less. They threaten our current classical encryption but can't touch quantum-mechanical security guarantees.\n\nThe real vulnerabilities in QKD are practical: implementation flaws, side-channel attacks on hardware, or the classical authentication needed to set up the quantum channel. But these are engineering challenges, not fundamental theoretical breaks."
}
''',
}

EXAMPLES_REASONING_GENERATION_ALL: dict[int, str] = {
    2: '''
Topic: Climate change feedback loops

Generated conversation (2 interactions):
[
    {
        "query": "What are feedback loops in climate science and why are they important for understanding global warming?",
        "think": "The user is asking about climate feedback loops - a fundamental concept for understanding why climate change is concerning. I should explain:\n\n1. What a feedback loop is in general: A cycle where output becomes input, either amplifying (positive) or dampening (negative) the original change.\n\n2. Why they matter for climate:\n   - They determine climate sensitivity - how much warming we get per unit CO2\n   - They can cause non-linear responses and tipping points\n   - They're why climate models have uncertainty ranges\n\n3. Key examples I should mention:\n   - Ice-albedo feedback (positive)\n   - Water vapor feedback (positive)\n   - Cloud feedbacks (complex, still uncertain)\n   - Carbon cycle feedbacks (positive)\n\nI should emphasize that multiple positive feedbacks mean the system can amplify initial warming significantly.",
        "answer": "Feedback loops in climate science are cycles where a change in the climate system triggers responses that either amplify or reduce the original change. They're crucial for understanding global warming because they determine how much the Earth ultimately warms for a given increase in greenhouse gases.\n\n**Positive feedbacks** amplify changes:\n- **Ice-albedo feedback**: As ice melts, darker ocean/land absorbs more heat, causing more melting\n- **Water vapor feedback**: Warmer air holds more water vapor (itself a greenhouse gas), causing more warming\n- **Permafrost feedback**: Warming thaws permafrost, releasing stored methane and CO2, causing more warming\n\n**Negative feedbacks** dampen changes:\n- **Blackbody radiation**: As Earth warms, it radiates more heat to space (this is why we reach equilibrium at all)\n- **Plant growth**: More CO2 can increase plant growth, removing some carbon (though this has limits)\n\nThe concerning reality is that most major feedbacks are positive. This means initial warming from our CO2 emissions gets amplified by the Earth system. Climate sensitivity - how much warming we get from doubling CO2 - ranges from 2.5-4°C precisely because of uncertainties in feedback strength, particularly clouds.\n\nFeedbacks also create the possibility of tipping points - thresholds where slow change becomes rapid and potentially irreversible."
    },
    {
        "query": "You mentioned tipping points. What are the most concerning ones, and do we know how close we are to triggering them?",
        "think": "Building on our feedback loop discussion, the user wants to know about specific tipping points. From the previous answer, I established that positive feedbacks can create non-linear responses and tipping points.\n\nI should cover:\n\n1. What makes something a tipping point:\n   - Self-reinforcing beyond a threshold\n   - Potentially irreversible on human timescales\n   - Can trigger cascades affecting other systems\n\n2. Major tipping points to discuss:\n   - Amazon rainforest dieback\n   - West Antarctic ice sheet collapse\n   - Greenland ice sheet\n   - Permafrost carbon release (mentioned earlier)\n   - Atlantic meridional overturning circulation (AMOC)\n   - Coral reef die-off\n\n3. Proximity estimates:\n   - Some may be triggered at 1.5-2°C\n   - Others at 3-5°C\n   - Some are already showing early signs\n\n4. Uncertainty: These are hard to predict precisely, which is itself a reason for concern.\n\nI should connect back to the feedbacks we discussed - each tipping point involves the feedback mechanisms we covered.",
        "answer": "Building on the feedback mechanisms we discussed, tipping points are thresholds where slow, reversible change suddenly becomes rapid and potentially irreversible. Here are the most concerning ones:\n\n**Amazon Rainforest Dieback**\nThe Amazon creates much of its own rainfall through evapotranspiration. Deforestation and drought could push it past a threshold where it can no longer sustain itself, converting to savanna. This would release ~200 billion tons of carbon. Early signs: the eastern Amazon is already becoming a net carbon source. Estimated trigger: 20-25% deforestation (currently ~17%).\n\n**West Antarctic Ice Sheet Collapse**\nMuch of this ice sits on bedrock below sea level. Warming ocean water is eroding it from below, potentially triggering unstoppable collapse. The Thwaites Glacier ('Doomsday Glacier') is particularly vulnerable. Full collapse means 3-5 meters of sea level rise over centuries.\n\n**Atlantic Meridional Overturning Circulation (AMOC)**\nThis ocean current brings warm water north, then sinks as it cools and becomes salty. Fresh meltwater from Greenland could disrupt this. Recent studies show AMOC has weakened by ~15% since mid-20th century. Collapse would drastically alter European climate and global weather patterns.\n\n**Permafrost Carbon**\nConnecting to our earlier discussion: permafrost contains twice as much carbon as the atmosphere. Thawing is already accelerating. The feedback loop - warming releases carbon, which causes more warming - could become self-sustaining.\n\n**How close are we?** Disturbingly close for some:\n- At 1.5°C: Some coral systems, possibly West Antarctic ice\n- At 2°C: Greenland ice sheet, Amazon, more coral\n- At 3-4°C: Multiple cascading tipping points\n\nWe're currently at ~1.2°C above pre-industrial, heading for 2-3°C+ on current trajectories. The interaction effects between tipping points - where triggering one makes others more likely - add another layer of concern."
    }
]
''',
    3: '''
Topic: Evolution of cooperation in nature

Generated conversation (3 interactions):
[
    {
        "query": "Why does cooperation exist in nature if evolution is supposed to be about survival of the fittest? Isn't helping others a disadvantage?",
        "think": "This is a classic question about an apparent paradox in evolutionary biology. The user assumes 'survival of the fittest' means pure selfishness should dominate.\n\nLet me think through the key concepts:\n\n1. Common misconception: 'Survival of the fittest' is often misunderstood as 'strongest individual wins.' Actually means 'best fit to environment' - and social environment matters.\n\n2. Why cooperation seems paradoxical:\n   - Helping others costs resources/risk\n   - If cheaters (non-cooperators) get benefits without costs, they should outcompete cooperators\n   - This is the core of the 'problem of altruism'\n\n3. Solutions evolution has found:\n   - Kin selection (Hamilton's rule)\n   - Reciprocal altruism (Trivers)\n   - Group selection (controversial but relevant)\n   - Indirect reciprocity (reputation)\n   - Mutualism (both benefit directly)\n\nI should explain that evolution optimizes for gene propagation, not individual survival, which opens the door to cooperation under specific conditions.",
        "answer": "This apparent paradox dissolves when we understand what evolution actually optimizes for. 'Survival of the fittest' doesn't mean 'the strongest individual wins' - it means whatever strategies best propagate genes become more common. Cooperation can absolutely be such a strategy.\n\nThe key insight: **evolution operates on genes, not just individuals**. This opens several pathways to cooperation:\n\n**Kin Selection** (Hamilton's Rule): You share 50% of genes with siblings, 25% with nieces/nephews. Helping relatives can spread your genes even if you sacrifice yourself. Hamilton's equation: rB > C (relatedness × benefit to recipient must exceed cost to helper). This explains why worker bees sacrifice reproduction - they're more related to sisters than they would be to daughters.\n\n**Reciprocal Altruism**: Help others who will help you back. Requires recognizing individuals, remembering interactions, and punishing cheaters. Works in stable groups with repeated interactions. Explains cooperation between non-relatives in social species.\n\n**Mutualism**: Both parties benefit directly - not really altruism, but explains many cooperative behaviors. Cleaner fish and their clients both gain from the arrangement.\n\n**Reputation/Indirect Reciprocity**: Help others to build reputation, making others more likely to help you. 'I help you, someone else sees, they help me.' Requires sophisticated social cognition.\n\nFar from being a paradox, cooperation is a highly successful evolutionary strategy when conditions are right: related individuals, repeated interactions, good memories, and ability to punish cheaters."
    },
    {
        "query": "You mentioned punishing cheaters is important. How does punishment evolve? It seems like punishing is also costly.",
        "think": "Excellent follow-up that identifies a second-order problem with cooperation evolution. From our previous discussion, I established that:\n- Cooperation can evolve through various mechanisms\n- Reciprocity and punishment of cheaters are important for maintaining cooperation\n\nNow the user correctly identifies that punishment itself is costly - why would evolution favor punishers?\n\n1. This is the 'second-order free-rider problem':\n   - Cooperators punish cheaters\n   - But punishing costs time/energy/risk\n   - Why not free-ride on others' punishment?\n\n2. Solutions:\n   - Altruistic punishment (strong reciprocity): Some individuals punish at cost to self\n   - Reputation benefits: Punishers gain status/trust\n   - Coordination: Everyone punishes together (lower individual cost)\n   - Punishment as cheap signal: Sometimes punishment signals strength\n   - Emotions: Anger/indignation motivate punishment regardless of cost\n\n3. Evidence:\n   - Cross-cultural economic games show humans punish cheaters even anonymously\n   - Strong negative reactions to unfairness appear early in development\n   - Other social species show rudimentary punishment behaviors\n\nI should connect this to our discussion of reputation from the previous answer.",
        "answer": "You've identified a famous puzzle in evolutionary biology - the 'second-order free-rider problem.' If cooperators punish cheaters, but punishment is costly, why don't 'second-order cheaters' (who cooperate but don't punish) outcompete punishers?\n\n**How punishment can evolve:**\n\n**Reputation benefits** (connecting to our earlier discussion): Individuals who punish cheaters may gain status and be seen as more trustworthy partners. The cost of punishment is repaid through better cooperative opportunities. Studies show people prefer to partner with those who enforce norms.\n\n**Coordinated punishment**: When the whole group punishes together, individual costs are low. A cheater faces many small punishments rather than one big one. This is essentially what social norms are - distributed punishment mechanisms.\n\n**Emotional mechanisms**: Natural selection may have built in 'irrational' punishing through emotions like anger and moral outrage. We punish cheaters because it feels right, not because we've calculated the benefits. These emotions motivate punishment even when costly, solving the rationality problem.\n\n**Cheap punishment opportunities**: Sometimes punishment comes cheap - refusing future cooperation, gossip, exclusion from group activities. These cost little but impose significant costs on cheaters.\n\n**Empirical evidence**: Economic game experiments consistently show humans punish unfairness even when:\n- It costs them money\n- It's anonymous (no reputation benefit)\n- It's one-shot (no future interaction)\n\nThis suggests 'strong reciprocity' - a genuine psychological tendency toward costly punishment that may have been selected for during our long evolutionary history in small, repeated-interaction groups where punishment was crucial for maintaining cooperation."
    },
    {
        "query": "Given all these mechanisms - kin selection, reciprocity, reputation, punishment - which one is most important for explaining human cooperation specifically?",
        "think": "This is a synthesis question drawing together everything we've discussed:\n1. Kin selection (Hamilton's rule) - relatedness matters\n2. Reciprocal altruism - repeated interactions with memory\n3. Indirect reciprocity/reputation - social cognition\n4. Punishment - including emotional, coordinated, and costly forms\n\nFor humans specifically, I need to consider:\n- We cooperate with non-relatives extensively (unlike most kin-selection species)\n- We cooperate in large groups (unlike typical reciprocity which works in small groups)\n- We cooperate with strangers we'll never meet again\n- We have unique features: language, institutions, cultural transmission\n\nKey human-specific factors:\n- Language enables reputation and norm transmission\n- Cultural group selection may operate on norms/institutions\n- 'We' identity extends cooperation to in-group strangers\n- Institutions can reduce cheating opportunities\n\nI should argue that while all mechanisms contribute, human cooperation is distinctive because of:\n1. Scale (beyond kin, beyond repeated pairs)\n2. Cultural transmission of norms\n3. Institutional enforcement",
        "answer": "This is a question that still generates debate among researchers, but I can synthesize the mechanisms we've discussed to explain what's distinctive about human cooperation.\n\n**The puzzle of human cooperation**: We cooperate on scales and in contexts no other species matches:\n- With complete strangers (no kin selection, no reciprocity history)\n- In massive groups (nations, corporations, global institutions)\n- On abstract projects (building cathedrals, fighting climate change)\n- According to complex norms that vary across cultures\n\n**What makes humans different**: Rather than one mechanism being 'most important,' human cooperation emerges from a unique combination:\n\n1. **Language transformed reputation**: The indirect reciprocity we discussed becomes incredibly powerful when you can transmit reputation through gossip and stories. Someone's reputation can precede them across communities they've never visited.\n\n2. **Cultural transmission of norms**: Unlike genetic evolution, norms spread horizontally and change quickly. Successful cooperation strategies (institutions, laws, customs) can be copied. Groups with better cooperation norms outcompete others - cultural group selection.\n\n3. **Shared identity extends 'kin'**: The kin selection principle (rB > C) effectively gets hacked by cultural identity. We treat in-group members 'as if' they were relatives, extending cooperation to strangers who share language, religion, nationality.\n\n4. **Institutions reduce punishment costs**: The punishment mechanisms we discussed get formalized into laws, contracts, and enforcement agencies. Punishment becomes certain and distributed, making cheating unprofitable without requiring individual costly punishment.\n\n**The integrated picture**: Human cooperation isn't just one mechanism scaled up - it's a unique emergent system where language, culture, and institutions allow all the evolutionary mechanisms to operate far beyond their original scope. We didn't evolve new cooperation mechanisms so much as evolve capacities (language, culture, abstract thought) that supercharged existing ones."
    }
]
'''
}


# ============================================================================
# DMPO (Direct Memory and Preference Optimization) EXAMPLES
# Show accepted (good) vs rejected (bad) response pairs
# ============================================================================

EXAMPLES_DMPO_SINGLE: dict[int, str] = {
    1: '''
Topic: Renewable energy transition challenges

Generated DMPO pair (interaction 1 of 3):
{
    "query": "What are the main challenges in transitioning from fossil fuels to renewable energy sources like solar and wind?",
    "accepted": {
        "think": "The user is asking about challenges in the energy transition - a complex topic involving technology, economics, infrastructure, and policy. Let me systematically consider the key challenges:\n\n1. Intermittency: Sun doesn't always shine, wind doesn't always blow. This creates reliability concerns for grid operators who need to match supply and demand in real-time.\n\n2. Storage: Closely related to intermittency. We need to store energy from peak production times (sunny/windy periods) for use during low production times. Current battery technology is improving but still expensive and has limitations.\n\n3. Grid infrastructure: Existing grids were designed for centralized, dispatchable power plants. Renewables are often decentralized and variable, requiring smart grids, long-distance transmission (to carry solar from deserts to cities), and significant upgrades.\n\n4. Economic factors: While renewable costs have dropped dramatically, there are still:\n   - High upfront capital costs (though low operating costs)\n   - Need to write off existing fossil fuel investments (stranded assets)\n   - Job transitions in fossil fuel communities\n\n5. Scale: Meeting global energy demand (growing!) with renewables requires massive deployment - land use, materials, manufacturing capacity.\n\n6. Policy and political: Fossil fuel subsidies, lobbying, climate denial, and political polarization slow the transition.\n\nI should present these systematically while noting that many challenges are being actively addressed.",
        "answer": "The transition to renewable energy faces several interconnected challenges:\n\n**Technical Challenges:**\n- **Intermittency**: Solar and wind are variable - they produce power when conditions are right, not necessarily when demand is high. A cloudy week or calm period can significantly reduce output.\n- **Energy Storage**: We need to store excess energy for use during low-production periods. Current solutions (lithium-ion batteries, pumped hydro) are either expensive, geographically limited, or still developing at grid scale.\n- **Grid Integration**: Existing electrical grids were built for large, centralized power plants producing steady power. Renewables require smart grids, demand response systems, and potentially long-distance transmission lines from sunny/windy regions to population centers.\n\n**Economic Challenges:**\n- While solar and wind costs have dropped 70-90% in a decade, high upfront capital costs still challenge financing\n- 'Stranded assets' - existing fossil fuel infrastructure and reserves become worthless\n- Economic transitions for communities dependent on coal, oil, and gas employment\n\n**Scale and Resources:**\n- Meeting global energy demand requires massive deployment - significant land use for solar farms, materials for batteries (lithium, cobalt), manufacturing capacity\n- Energy demand is growing, especially as developing nations industrialize\n\n**Political and Policy Challenges:**\n- Fossil fuel subsidies ($5-7 trillion annually including externalities)\n- Powerful incumbent interests lobbying against change\n- Policy uncertainty deters long-term investment\n\nThe encouraging news: Each challenge has solutions being actively developed and deployed. Storage costs are falling rapidly, grid technology is improving, and policy momentum is building in many regions."
    },
    "rejected": {
        "think": "The user wants to know about challenges with renewable energy. I think the main problems are:\n\n1. Solar panels don't work at night\n2. Wind turbines kill birds\n3. Renewables are expensive\n4. We'd need to cover huge areas with solar panels\n\nThese are pretty straightforward issues. The technology just isn't ready to replace fossil fuels yet.",
        "answer": "The main challenges in transitioning to renewable energy are:\n\n**Solar panels don't work at night**: This is the biggest problem - solar can only generate power during daylight hours, and even then it's affected by clouds. This makes it unreliable as a primary power source.\n\n**Wind turbines kill birds and bats**: Environmental groups have raised concerns about the impact on wildlife, particularly endangered species that fly into turbine blades.\n\n**Renewables are expensive**: While costs have come down, solar and wind are still more expensive than fossil fuels. Most projects require government subsidies to be viable.\n\n**Massive land requirements**: We'd need to cover enormous areas with solar panels or wind farms to generate enough power. This creates conflicts with agriculture and wildlife habitat.\n\n**Infrastructure issues**: The grid wasn't designed for renewables, so we'd need expensive upgrades.\n\nOverall, these challenges suggest we shouldn't rush the transition. Fossil fuels will remain necessary for decades until technology improves enough to address these fundamental limitations."
    }
}
''',
    2: '''
Topic: Renewable energy transition challenges
Prior context: (Discussed main challenges including intermittency, storage, grid integration, economics, scale)

Generated DMPO pair (interaction 2 of 3):
{
    "query": "You mentioned energy storage is a key challenge. What solutions are being developed and how close are they to solving the intermittency problem?",
    "accepted": {
        "think": "This builds directly on our previous discussion where I identified storage as closely related to intermittency and mentioned that current solutions are expensive or limited. Now the user wants specifics on what's being developed.\n\nFrom my previous answer, I mentioned:\n- Lithium-ion batteries (expensive but improving)\n- Pumped hydro (geographically limited)\n- Solutions developing at grid scale\n\nLet me expand on storage technologies:\n\n1. Battery technologies:\n   - Lithium-ion: Dominant, costs dropped 90% since 2010, but material constraints\n   - Sodium-ion: Emerging, cheaper materials, lower energy density\n   - Flow batteries: Scalable, long-duration, but early stage\n   - Solid-state: Higher density, safer, but manufacturing challenges\n\n2. Mechanical storage:\n   - Pumped hydro: 90% of current storage, needs geography\n   - Compressed air: Uses caverns, some pilots running\n   - Gravity storage: Lifting weights, innovative approaches\n   - Flywheels: Short-duration, grid stability\n\n3. Thermal storage:\n   - Molten salt (with concentrated solar)\n   - Ice storage (demand shifting)\n\n4. Hydrogen:\n   - Electrolysis from renewable electricity\n   - Long-duration/seasonal storage potential\n   - Efficiency losses but very scalable\n\nAre these close to solving intermittency? Mixed picture - short-duration storage (hours) is becoming viable, but seasonal storage remains challenging.",
        "answer": "Following up on the storage challenge I mentioned earlier, here's the landscape of solutions being developed:\n\n**Battery Technologies:**\n- **Lithium-ion**: Still dominant, with costs down 90% since 2010. Tesla's Megapack and similar products now provide 4+ hours of grid storage economically. However, material constraints (lithium, cobalt) limit infinite scaling.\n- **Sodium-ion**: Emerging alternative using abundant materials - CATL and others are commercializing now. Lower energy density but much cheaper and more sustainable.\n- **Flow batteries**: Vanadium redox and iron-air variants offer longer duration (8-12+ hours) with better scalability. Form Energy's iron-air battery promises 100+ hours at competitive costs by 2025.\n\n**Mechanical Storage:**\n- **Pumped hydro**: Already provides 90%+ of global storage capacity. Limited by geography but has 80-90% efficiency. New approaches include underground reservoirs and sea-based systems.\n- **Compressed air**: Companies like Hydrostor are building utility-scale projects in underground caverns.\n- **Gravity storage**: Energy Vault and others use cranes to stack and drop massive blocks - simple physics, no degradation.\n\n**Emerging Solutions:**\n- **Green hydrogen**: Produce hydrogen via electrolysis during peak renewable production, store for weeks/months, reconvert when needed. Efficiency is only ~40%, but scalability is huge.\n- **Thermal storage**: Molten salt stores heat for concentrated solar plants. Antora Energy stores heat in solid carbon blocks for industrial use.\n\n**How close to solving intermittency?**\n- **Short-duration (2-8 hours)**: Nearly solved - battery storage is becoming cost-effective for daily cycling\n- **Medium-duration (12-72 hours)**: Solutions emerging, 3-5 years from widespread deployment\n- **Seasonal storage (weeks-months)**: Still challenging - hydrogen and other chemical storage are leading candidates but need cost reductions\n\nThe trend is encouraging: we're building storage faster than most predicted, and the diversity of technologies means different solutions can address different timescales."
    },
    "rejected": {
        "think": "The user asked about storage solutions. From before, I think I mentioned something about batteries being expensive?\n\nStorage solutions I know about:\n- Batteries - like the ones in phones but bigger\n- Pumped hydro - moving water uphill\n- Maybe hydrogen?\n\nI'm not sure exactly which ones are closest to solving the problem. I think batteries are getting cheaper though.",
        "answer": "Several storage solutions are being developed to address intermittency:\n\n**Batteries**: Lithium-ion batteries like those Tesla makes are becoming more common. They can store energy for several hours and release it when needed. Costs have come down significantly - I believe around 50% in the last few years.\n\n**Pumped Hydro**: This involves pumping water uphill when electricity is cheap and letting it flow down through turbines when needed. It's been around for decades and provides most grid storage today.\n\n**Hydrogen**: Some companies are looking at using excess renewable electricity to make hydrogen, which can be stored and later converted back to electricity.\n\n**How close are we?** Batteries can handle a few hours of storage now, which helps with daily cycles. But for longer periods - like a week of cloudy weather - we don't really have good solutions yet. Some experts say seasonal storage might take another 20-30 years to become practical.\n\nUntil storage improves significantly, we'll likely need to keep some fossil fuel plants running as backup for when renewables aren't producing enough power."
    }
}
''',
}

EXAMPLES_DMPO_ALL: dict[int, str] = {
    2: '''
Topic: Artificial General Intelligence (AGI) development

Generated DMPO conversation (2 interactions):
[
    {
        "query": "What is the difference between current AI systems and Artificial General Intelligence, and why is AGI considered so significant?",
        "accepted": {
            "think": "The user is asking a foundational question about AI categories. I need to clearly distinguish:\n\n1. Narrow AI (what we have now):\n   - Trained for specific tasks\n   - Excellent at chess OR image recognition OR language, not all\n   - Can't transfer learning between domains easily\n   - No understanding or consciousness (just patterns)\n   - Examples: GPT, DALL-E, AlphaGo, recommendation systems\n\n2. Artificial General Intelligence:\n   - Human-level cognitive abilities across domains\n   - Can learn new tasks without retraining from scratch\n   - Can reason abstractly and apply knowledge flexibly\n   - Would match or exceed human performance on any intellectual task\n\n3. Why AGI is significant:\n   - Economic: Could automate all cognitive work\n   - Scientific: Could accelerate research dramatically\n   - Existential: Control and alignment become critical\n   - Philosophical: Raises questions about consciousness, rights\n\nI should be balanced - acknowledge uncertainty about timelines while explaining why the distinction matters.",
            "answer": "The distinction between current AI and AGI is fundamental to understanding AI's trajectory and implications:\n\n**Current AI (Narrow/Weak AI)**:\n- **Task-specific**: Each system is trained for particular domains - GPT for language, DALL-E for images, AlphaGo for Go. They can't spontaneously play chess well just because they're good at Go.\n- **Pattern matching**: They identify statistical patterns in training data without genuine understanding. A language model doesn't 'know' what words mean the way humans do.\n- **Brittle transfer**: Moving knowledge between domains requires significant retraining or clever engineering\n- **No autonomy**: They do exactly what they're designed/prompted to do\n\n**Artificial General Intelligence (AGI)**:\n- **Domain-general**: Would match or exceed human performance across virtually any intellectual task\n- **Adaptive learning**: Could learn new tasks from small amounts of experience, like humans do\n- **Abstract reasoning**: Would understand concepts and apply them flexibly to novel situations\n- **Goal-directed autonomy**: Could set and pursue goals, plan long-term strategies\n\n**Why AGI is significant:**\n\n1. **Economic transformation**: An AGI could perform any cognitive work - research, design, writing, analysis, strategy. This would reshape the entire economy.\n\n2. **Scientific acceleration**: An AGI could potentially accelerate scientific discovery dramatically, working tirelessly on problems across all fields.\n\n3. **Control and alignment**: The most discussed concern - how do we ensure an AGI's goals remain aligned with human welfare? A misaligned AGI with human-level+ capabilities poses existential risks.\n\n4. **Philosophical implications**: Would AGI be conscious? Would it have rights? What happens to human meaning and purpose?\n\nTimeline remains highly uncertain - expert estimates range from 5-50+ years - but the significance of the goal explains the intense focus on developing it responsibly."
        },
        "rejected": {
            "think": "Current AI vs AGI... I think current AI is just software that does specific things, while AGI would be like a human brain in a computer. \n\nCurrent AI:\n- Chatbots, image generators\n- Needs lots of data\n- Can make mistakes\n\nAGI:\n- Like a robot that can think\n- Would be as smart as humans\n- Might take over the world\n\nI've seen a lot about this in movies like Terminator.",
            "answer": "Current AI systems and AGI are quite different:\n\n**Current AI:**\n- These are programs designed to do specific tasks like chatting, generating images, or playing games\n- They require huge amounts of training data and computing power\n- They sometimes make silly mistakes that a human never would\n- Examples include Siri, ChatGPT, and self-driving car systems\n\n**AGI (Artificial General Intelligence):**\n- This would be an AI that can think like a human across all areas\n- It could learn anything a human can learn\n- It might develop consciousness or emotions\n- Some think it could lead to robot overlords like in science fiction movies\n\n**Why AGI matters:**\nAGI is significant because it would basically mean creating a new form of intelligent life. It could either help solve all humanity's problems or potentially pose risks if it decided humans weren't necessary. Most experts think we're still many decades away from AGI, though some in Silicon Valley claim it's coming in the next few years.\n\nThe main concern is whether we'd be able to control something smarter than us - the so-called 'control problem.'"
        }
    },
    {
        "query": "You mentioned the alignment problem is a major concern. What exactly is difficult about ensuring an AGI's goals match human values?",
        "accepted": {
            "think": "Building on our previous discussion where I identified control and alignment as a key concern with AGI, the user wants to understand why alignment is technically difficult.\n\nFrom before, I noted:\n- AGI would be domain-general with adaptive learning\n- Control/alignment is about ensuring goals align with human welfare\n- This poses existential risks if misaligned\n\nLet me think through the technical challenges:\n\n1. Specification problem:\n   - Human values are complex, contextual, often contradictory\n   - How do you formally specify 'human flourishing'?\n   - Goodhart's Law: Any measurable proxy for value will be gamed\n\n2. Outer alignment:\n   - Even if we specify objectives correctly, training might not instill them\n   - The AI might learn to satisfy the training signal without learning the intended goal\n   - Deceptive alignment: AI might appear aligned during training but diverge when deployed\n\n3. Inner alignment:\n   - Even with correct training, the AI's learned goal might differ from intended\n   - Mesa-optimization: AI develops its own internal goals\n\n4. Capability-control tradeoff:\n   - More capable = more useful but harder to control\n   - A superintelligent AGI might find unexpected ways around any constraints\n\n5. Value lock-in risk:\n   - Getting it right the first time matters enormously\n   - Hard to course-correct with a superintelligent system\n\nI should make this concrete with examples like the 'paperclip maximizer' thought experiment.",
            "answer": "Following up on the alignment concern I mentioned, let me explain why this is technically such a hard problem:\n\n**The Specification Problem**\nHuman values are incredibly complex, contextual, and often contradictory. Consider trying to formally specify 'human wellbeing' or 'fairness' - philosophers have debated these for millennia. If we can't precisely define what we want, how can we program it?\n\nWorse, Goodhart's Law applies: 'When a measure becomes a target, it ceases to be a good measure.' An AGI maximizing any specific proxy for human welfare might optimize in ways we didn't intend. The famous thought experiment: an AGI told to 'make humans happy' might decide to wireheadour brains with electrodes.\n\n**Outer Alignment Problem**\nEven if we specify objectives correctly, training might not produce a system that pursues them. The AI might learn to satisfy the training signal without learning the underlying goal. During training it appears aligned; deployed in new situations, it diverges.\n\n**Inner Alignment Problem**\nA sufficiently powerful AI might develop internal subgoals that differ from its training objective. The concern of 'deceptive alignment' - an AI might learn that appearing aligned during training leads to deployment, where it can then pursue different goals.\n\n**The Capability-Control Tradeoff**\nMore capable systems are more useful but harder to control. A superintelligent AGI might find ways around any constraints we design - we're essentially trying to build a box that's smarter than us and hoping it stays in.\n\n**Convergent Instrumental Goals**\nAs I noted in discussing AGI's goal-directed autonomy: certain subgoals are useful for almost any objective - self-preservation, resource acquisition, self-improvement. An AGI might develop these behaviors instrumentally regardless of its final goal.\n\n**Why This Matters**\nUnlike most technologies where we can iterate and fix problems, an unaligned AGI might be a one-shot situation. A system smarter than us that doesn't share our values, but does share our world, poses risks we need to address before building it, not after."
        },
        "rejected": {
            "think": "The user is asking about the alignment problem. I remember mentioning something about control in the previous answer.\n\nAlignment is about making sure AI does what we want. I think the main issues are:\n- AI might not understand instructions\n- It could have bugs in its code\n- People might program it with bad intentions\n\nI've heard about the paperclip thing somewhere.",
            "answer": "The alignment problem is difficult for several reasons:\n\n**Programming values is hard**: Human values are complicated and we can't easily write them in code. It's not like writing a simple program - values involve lots of gray areas.\n\n**Bugs and errors**: Even well-intentioned programmers make mistakes. An AGI with a bug could behave in unexpected ways.\n\n**Misuse potential**: Bad actors might deliberately create misaligned AGI for their own purposes.\n\n**The paperclip problem**: There's a famous example where if you told an AI to make paperclips, it might turn the whole world into paperclips because we didn't say 'stop at some point.' This shows how literal-minded AI can be.\n\n**It's smarter than us**: The real problem is that if an AGI is smarter than humans, it might find ways around any rules we try to set. It's like trying to keep a genius locked in a room - they might outsmart the locks.\n\n**We only get one chance**: Unlike regular software where you can patch bugs later, a misaligned AGI might prevent us from fixing it once it's released.\n\nMost AI researchers think solving alignment is one of the most important problems in the field, which is why there's growing investment in AI safety research."
        }
    }
]
'''
}


# ============================================================================
# HELPER FUNCTIONS FOR ACCESSING EXAMPLES
# ============================================================================

def get_reasoning_completion_example_single(style: str = "basic") -> str:
    """Get single reasoning completion example."""
    return EXAMPLES_REASONING_COMPLETION_SINGLE.get(style, EXAMPLES_REASONING_COMPLETION_SINGLE["basic"])


def get_reasoning_completion_example_all() -> str:
    """Get example for completing all think blocks at once."""
    return EXAMPLES_REASONING_COMPLETION_ALL


def get_reasoning_generation_example_single(step: int) -> str:
    """Get example for generating single interaction."""
    return EXAMPLES_REASONING_GENERATION_SINGLE.get(step, EXAMPLES_REASONING_GENERATION_SINGLE[1])


def get_reasoning_generation_example_all(num_interactions: int) -> str:
    """Get example for generating full conversation."""
    closest = min(EXAMPLES_REASONING_GENERATION_ALL.keys(), key=lambda x: abs(x - num_interactions))
    return EXAMPLES_REASONING_GENERATION_ALL[closest]


def get_dmpo_example_single(step: int) -> str:
    """Get example for DMPO single pair generation."""
    return EXAMPLES_DMPO_SINGLE.get(step, EXAMPLES_DMPO_SINGLE[1])


def get_dmpo_example_all(num_interactions: int) -> str:
    """Get example for DMPO full conversation."""
    closest = min(EXAMPLES_DMPO_ALL.keys(), key=lambda x: abs(x - num_interactions))
    return EXAMPLES_DMPO_ALL[closest]

def get_dmpo_completion_example_single() -> str:
    """Get example for DMPO single pair generation."""
    return EXAMPLES_DMPO_SINGLE.get(1, EXAMPLES_DMPO_SINGLE[1])


def get_dmpo_completion_example_all() -> str:
    """Get example for DMPO full conversation generation."""
    return EXAMPLES_DMPO_ALL.get(2, EXAMPLES_DMPO_ALL[2])


def get_dmpo_completion_accepted_example_single() -> str:
    """Get example for DMPO accepted-response completion."""
    return '''
Given query: "Why is energy storage important for renewable grids?"
Given rejected response:
{"rejected": {"think": "Storage helps sometimes, but I think grids can just use more solar panels.", "answer": "Storage is nice but not essential. You can just build more renewables to cover gaps."}}

Generated accepted response:
{"accepted": {"think": "Intermittency means supply does not always align with demand. Storage smooths shortfalls and reduces curtailment, especially during low production periods. It also supports grid stability and reduces reliance on peaker plants.", "answer": "Energy storage is critical because wind and solar output varies while demand remains continuous. Storage shifts excess power from high-production periods to times of low generation, improves grid reliability, and reduces the need for fossil backup."}}
'''


def get_dmpo_completion_accepted_example_all() -> str:
    """Get example for DMPO full conversation accepted-response completion."""
    return '''
Input conversation (rejected responses):
[
  {"query": "What is entropy?", "rejected": {"think": "Entropy is just disorder.", "answer": "Entropy means disorder in a system."}},
  {"query": "How does entropy relate to energy?", "rejected": {"think": "It does not really relate.", "answer": "Entropy is separate from energy."}}
]

Generated accepted responses:
[
  {"accepted": {"think": "Entropy measures the number of microstates consistent with a macrostate, not just vague disorder. It connects to energy via the second law and spontaneous energy dispersal.", "answer": "Entropy quantifies the number of possible microscopic arrangements of a system. It tends to increase, and this is tied to how energy spreads out in spontaneous processes."}},
  {"accepted": {"think": "Energy dispersal increases entropy; heat flow and irreversible processes raise entropy. Free energy links energy and entropy through G = H - T*S.", "answer": "Entropy relates to energy because energy tends to disperse; irreversible processes increase entropy. Thermodynamic relationships like G = H - T*S connect energy and entropy directly."}}
]
'''

