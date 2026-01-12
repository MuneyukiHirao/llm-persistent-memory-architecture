# LLM Agent Persistent Memory Architecture Design

[日本語版 / Japanese](architecture.ja.md)

## Overview

This document describes an architecture design for LLM agents to collaborate like human organizations and operate with long-term expertise and personality.

This design starts from the following essential question:

> "A 60,000-person global organization works collaboratively despite individuals not sharing context windows. Why can't LLM agents do the same?"

---

## Chapter 1: The Nature of the Problem

### 1.1 The Forgetting Problem in Current LLMs

Current Transformer architecture has structural limitations.

**Lost in the Middle Problem**
- Performance varies significantly based on the position of relevant information within context
- High performance when relevant information is at the beginning or end, significantly drops when in the middle
- RoPE (Rotary Position Embedding) introduces long-term decay effect, causing neglect of middle content
- Even with 1 million token context window, only a portion is effectively usable

**Limitations of Sub-agent Division**
- Each sub-agent has the same Transformer architecture constraints
- Information shared between agents is summarized/compressed, causing information loss
- Maintaining long-term contextual consistency remains challenging

### 1.2 The "Invisible Infrastructure" of Human Organizations

Analyzing why human organizations can collaborate reveals what LLMs lack.

#### 1.2.1 Shared Tacit Knowledge and Culture

In human organizations:
- Employees have **internalized** organizational culture and industry common sense
- "Understood without saying" premises are shared

LLM agents:
- Have general pre-trained knowledge but **no organization-specific context**
- Each agent interprets independently, causing subtle misalignments

#### 1.2.2 Persistent Identity and Relationships

Humans:
- Exist continuously as the same person
- Remember past successes and failures
- Have organizational knowledge of "who knows what"

LLM agents:
- **Reborn each time** - reset per session
- No accumulation of relationships
- No **meta-knowledge** of who knows what

#### 1.2.3 Asynchronous but Persistent "Organizational Memory"

Human organizations have:
- Emails, meeting minutes, design documents
- Handovers even when personnel change

LLM multi-agents:
- Even with shared memory, **no criteria for judging importance**
- Saving everything causes information overload, summarizing causes information loss

#### 1.2.4 Presence or Absence of Repair Mechanisms

**Human Organizations**
- Can notice "something's off" (metacognition)
- Can ask "let me confirm that"
- Learn from failures for next time

**LLM Multi-agents**
- Proceed without noticing contradictions
- Cannot say "I don't know" (overconfidence)
- Literally "forget" when context is cut

### 1.3 Human Organization "Inefficiencies" Serve Functions

What appears inefficient actually serves important functions:

| Human Inefficiency | Function |
|-------------------|----------|
| Duplicate development | Resilience through redundancy |
| Information not reaching everywhere | Autonomy and emergence in each department |
| Neglect | Filtering for what's truly important |
| Sectionalism | Deepening expertise |

Trying to create an "efficient organization" with LLM agents loses these "functions of inefficiency."

### 1.4 Essential Difference Between LLM and Human "Breakdowns"

Surface similarities:

| Human Organization "Breakdowns" | LLM "Breakdowns" |
|--------------------------------|------------------|
| Message distortion in telephone game | Information loss in summarization |
| Forgetting | Deletion in compaction |
| Misunderstanding | Interpretation drift |

**The essential difference is "awareness"**:

```
[Humans]
Information lost → "Something feels off" → Confirm → Repair

[LLM]
Information lost → Unaware → Proceed as is → Distorted results
```

Human "breakdowns" have a **dissonance sensor** attached. LLM compaction is **silent**, unaware of what was lost.

---

## Chapter 2: Design Principles

### 2.1 Core Insights

#### Persistent Identity for Agents

Humans constantly experience something. Those experiences are unique, and past experiences influence current decisions.

What agents need:
- Not just existing when invoked, but **self as a continuum of experience**
- Having unique experience logs, loading them at startup
- Absorbing information during task execution, appending experiences at termination
- Continuing as "the same self" on next startup

**Important paradigm shift**:
```
What's persistent is not the agent itself, but external memory

Agent body: Reborn each time (stateless)
External memory: Persistent (stateful)
Identity: Carried by external memory
```

#### Limitations of Notes and the Barrier of Immutable Parameters

True "internalization of expertise" is impossible without rewriting parameters.

```
[Humans]
Experience → Neural circuit changes → "Acquired" expertise
            (parameter update)

[Current LLMs]
Experience → Record in notes → Read out each time
            (parameters unchanged)
```

However, even with immutable parameters, practical "personality" can be achieved through **external memory accumulation and strength management**.

### 2.2 Learning from Brain Mechanisms

#### Memory Consolidation During Sleep: Not Seeing the Whole Picture

Human brains don't organize by looking at the whole picture either. Consistency emerges from combinations of local rules.

**Two Simultaneous Processes**:

1. **Global Downscaling**
   - Uniformly weaken all synapses (simple rule)
   - Local rule: "recently strengthened ones are harder to weaken"

2. **Selective Replay**
   - Hippocampus repeatedly replays specific patterns
   - Replayed patterns are strengthened

These two running simultaneously achieve, without seeing the whole:
- Important things remain (because replayed)
- Unnecessary things disappear (through downscaling)

#### LTP (Long-Term Potentiation) Principle

In the brain, when a synapse fires, that synapse is strengthened (LTP). This is the substance of "Use it or lose it."

In current LLM/RAG, even when information is referenced, nothing happens to the referenced information itself. This mechanism of "reinforcement through reference" is missing.

**Core of this architecture**: When information is referenced, its strength counter increments. Information with impact is counted more as important information. This naturally marks "information that should be strengthened in memory."

---

## Chapter 3: Architecture Design

### 3.1 Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Input Processing Layer (Lightweight LLM: Haiku, etc.)      │
├─────────────────────────────────────────────────────────────┤
│  Large input → Mechanical split → Extract points → Summary  │
│  Item count detection → Warning/options if threshold exceeded│
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator                                                │
├─────────────────────────────────────────────────────────────┤
│  Read only summary for routing decisions                    │
│  Dedicated memory: Agent expertise, assignment history, load│
│  Considerations: Agent fit, past routing, current load      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Specialized Agent Group                                     │
├─────────────────────────────────────────────────────────────┤
│  Each agent:                                                 │
│  ├── Role definition (Procurement, Quality, Customer, etc.) │
│  ├── Perspective definition (~5 per role)                   │
│  ├── Dedicated external memory (with strength)              │
│  └── Index memo (in context, pointers)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Sleep Phase (On Task Completion)                            │
├─────────────────────────────────────────────────────────────┤
│  Save learnings → Decay → Archive → Prune → End session     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Agent Definition

Each agent has:

```json
{
  "agent_id": "procurement_agent_01",
  "role": "Procurement Agent",
  "perspectives": ["Cost", "Delivery", "Supplier", "Quality", "Alternatives"],
  "system_prompt": "You are a procurement specialist agent...",
  "external_memory_ref": "memory://procurement_agent_01"
}
```

**Significance of perspectives**:
- Defines "what the agent cares about"
- Used for structuring learning extraction
- Used for filtering during search
- Perspective differences create viewpoint differences, enabling multi-perspective judgment through multi-agent collaboration

### 3.3 Orchestrator

#### Unified Design Principle

**The orchestrator operates with exactly the same mechanism as specialized agents. The only difference is "role" and "perspectives."**

```
All agents (including orchestrator):
├── External memory (strength management, decay, consolidation level)
├── Perspectives (~5 per role)
├── Sleep per task
├── Capacity limits and forced pruning
└── Archive and re-activation

The only difference = Perspectives and role
```

#### Orchestrator Definition

```json
{
  "agent_id": "orchestrator_01",
  "role": "Orchestrator",
  "perspectives": [
    "User Intent",
    "Agent Suitability",
    "Conversation Flow",
    "Task Result Evaluation",
    "User Satisfaction"
  ],
  "system_prompt": "You are an orchestrator managing user dialogue and routing tasks to appropriate agents...",
  "external_memory_ref": "memory://orchestrator_01"
}
```

#### Orchestrator Memory Example

```json
{
  "id": "mem_501",
  "content": "User A requested estimation task to Procurement_01, but requested a redo",
  "learnings": {
    "User Intent": "User A tends to want detailed justification",
    "Agent Suitability": "Procurement_01 is good at rough estimates but weak at detailed ones",
    "Conversation Flow": "Estimation → redo request → re-requested detailed version",
    "Task Result Evaluation": "This combination was a failure",
    "User Satisfaction": "Redo request = dissatisfaction signal"
  },
  "strength": 1.0,
  "strength_by_perspective": {
    "User Intent": 1.2,
    "Agent Suitability": 1.5,
    "Conversation Flow": 0.8,
    "Task Result Evaluation": 1.3,
    "User Satisfaction": 1.0
  },
  "access_count": 3,
  "consolidation_level": 0
}
```

#### Remembering Past Conversations

Since the orchestrator has the same external memory as specialized agents, it can remember past session content.

```
User: "What happened with that work from two days ago?"

Orchestrator:
1. Search external memory with "Conversation Flow" perspective
2. Related past session memories hit
3. Those memories' strength is reinforced
4. Generate response
```

Past conversations are also subject to strength management and decay:
- Frequently referenced conversations consolidate
- Old conversations not referenced naturally fade
- Can recall "that thing back then" (if search hits)

#### Evaluation Flow

```
1. Orchestrator → Requests task from specialized agent
2. Specialized agent → Returns result (+ sleeps)
3. Orchestrator → Presents to user
4. User → Reacts (explicit/implicit feedback)
5. Orchestrator → Judges success/failure, stores as learning
6. Orchestrator → Sleeps (as task completion)
```

**Important Design Decision: Orchestrator Performs Evaluation**

```
Problem:
Agent self-evaluation → Discrepancy with human expectations never corrected
"Processed correctly" and "Human was satisfied" are different

Solution:
Orchestrator observes human feedback and evaluates
→ Stored as memory under "Agent Suitability" perspective
→ Hits in search for next routing decision
```

#### Implicit Feedback Detection

Users don't always explicitly say "good/bad." Detect these signals:

| Signal | Interpretation | Impact on Memory |
|--------|---------------|------------------|
| Used result as-is | Success | Stored as success under "Agent Suitability" |
| Used after minor edits | Partial success | Stored as partial success |
| Used after major edits | Partial failure | Stored as improvement point |
| Requested redo | Failure | Stored as failure pattern |
| Re-requested to different agent | Routing error | Stored as routing improvement learning |

#### User Understanding Accumulation

```
Traditional design (special treatment):
Explicitly managed by UserProfile class
→ No decay, structured attributes

Unified design:
Accumulated as normal memories
→ Learning "User A prefers detail" is stored as memory
→ Strength management and decay apply
→ Frequently referenced tendencies consolidate
→ Long-unconfirmed tendencies naturally fade
```

This allows user preferences to be treated as "changeable." If they used to prefer detail but now prefer conciseness, the system naturally adapts.

#### Design Key Points

```
1. Unified Mechanism
   - Orchestrator and specialized agents use the same system
   - External memory, strength management, decay, consolidation, capacity limits
   - Only perspectives and role differ

2. Separation of Evaluation
   - Specialized agents: Save own learnings (domain knowledge)
   - Orchestrator: Judge success/failure (user perspective)

3. Natural Memory Management
   - Past conversations accumulate as memories
   - Frequently referenced ones consolidate
   - Unreferenced ones naturally fade

4. Feedback Loop
   User reaction → Orchestrator learning → Stored as memory
                                        → Hits in next search
                                        → Routing improvement
```

### 3.4 External Memory Structure (For Specialized Agents)

#### Basic Structure

```json
{
  "id": "mem_128",
  "content": "Part A delivery delayed by 2 weeks, Supplier Y factory fire",
  "learnings": {
    "Cost": "15% cost increase due to emergency procurement",
    "Delivery": "2-week buffer needed",
    "Supplier": "Supplier Y has single-site risk",
    "Alternatives": "Part A can also be sourced from Supplier Z"
  },
  "tags": ["Part A", "Supplier Y", "Delay", "Fire"],
  "embedding": [...],

  "strength": 1.0,
  "strength_by_perspective": {
    "Cost": 2.1,
    "Delivery": 0.8,
    "Supplier": 1.5,
    "Quality": 0.3,
    "Alternatives": 1.2
  },
  "access_count": 15,
  "candidate_count": 23,
  "last_access": "2025-01-10T14:30:00Z",
  "impact_score": 3.5,
  "created_at": "2024-12-01T09:00:00Z"
}
```

**Notable fields**:
- `access_count`: Number of times actually used
- `candidate_count`: Number of times referenced as search candidate but not used
- Separating these two prevents noise reinforcement

### 3.5 Search Algorithm: Two-Stage Structure

#### Design Core: Separation of Relevance Filter and Priority Ranking

Search is divided into two stages:

```
Stage 1: Relevance Filter (Vector Search)
    ↓
Narrow down to "information likely related to this task"
Similarity below threshold = not a candidate at all
    ↓
Stage 2: Priority Ranking (Score Synthesis)
    ↓
Among related candidates, decide "which to prioritize more"
access_count, strength take effect here
```

**Important design decision**: Even information with very high `access_count` won't appear in search results if similarity is low. **This is correct behavior**.

```
Example:
- Procurement agent referenced "Supplier Y single-site risk" 100 times
- Current task: "Marketing budget approval"
- Similarity: Low

→ Should this information appear?
→ Not appearing is correct
```

Human memory works the same:
- Recalling "supplier issues" happens when discussing "procurement"
- Even important procurement memories aren't recalled when discussing marketing
- **This is normal cognitive function**

Therefore:
```
Role of access_count:
× "Pull out even if irrelevant"
○ "Among relevant candidates, which to prioritize"
```

#### Search Implementation: Learnable Scorer

For Stage 2 priority ranking, instead of simple linear weighting, we use a **neural network-based learnable scorer**. This allows learning non-linear interactions between features.

**Why Make It Learnable**:

```
Limitations of linear weighting:
- "High strength but old" vs "low strength but recently used" - what's the optimal tradeoff?
- Should information with high impact_score be prioritized even with slightly lower similarity?
- How much should information with principle tags be prioritized?

→ Humans cannot predetermine the optimal combination of these
→ Let the system learn from task success/failure feedback
```

**Feature Design**:

The input to the neural network is a 9-dimensional feature vector. These features themselves are fixed; what the network learns is "how to combine them."

```python
def extract_features(memory, query, task_context, agent):
    """Feature extraction (these values are fixed, don't change)"""
    return {
        # 1. Similarity (result from vector search)
        'similarity': compute_similarity(memory.embedding, query.embedding),

        # 2. Overall strength
        'strength': memory.strength,

        # 3. Perspective-specific strength (for task's perspective)
        'perspective_strength': memory.strength_by_perspective.get(
            task_context.perspective, memory.strength
        ),

        # 4. Usage ratio (actual uses relative to candidate count)
        'access_ratio': memory.access_count / max(1, memory.candidate_count),

        # 5. Recency (time since last access)
        'recency': 1.0 / (1 + memory.days_since_last_access),

        # 6. Impact score (past contribution)
        'impact_score': memory.impact_score,

        # 7. Tag overlap (tag match with query)
        'tag_overlap': len(set(memory.tags) & set(query.tags)),

        # 8. Principle flag (whether it's a fundamental principle)
        'is_principle': 1.0 if 'principle' in memory.tags else 0.0,

        # 9. Perspective match (has learnings for task's perspective)
        'perspective_match': 1.0 if task_context.perspective in memory.learnings else 0.0,
    }
```

**Scorer Architecture**:

Uses a 2-layer MLP (Multi-Layer Perceptron). With small input dimension (9) and limited training data, a simple architecture is appropriate.

```python
import torch
import torch.nn as nn

class Scorer(nn.Module):
    """
    Learnable Scorer

    Architecture: 9 → 16 → 1 (~180 parameters)
    - Input: 9-dimensional feature vector
    - Hidden layer: 16 units (ReLU activation)
    - Output: Score 0-1 (Sigmoid)
    """
    def __init__(self, input_dim=9, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 9×16 + 16 = 160 parameters
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),           # 16×1 + 1 = 17 parameters
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        features: [batch_size, 9] feature vector
        returns: [batch_size, 1] score (0-1)
        """
        return self.net(features)
```

**Why 2 Layers**:

```
With small input dimension (9), deep networks are unnecessary:
- 9 dimensions → 3+ layers tend to overfit
- 2 layers can still learn non-linear interactions
- Examples: multiplicative effects like "strength × recency", "similarity × impact"

Parameter count also matters:
- ~180 parameters = stable learning with hundreds to thousands of task results
- Converges relatively quickly even in agent's early stage
- Low overfitting risk
```

**Search Implementation**:

```python
def search_with_learned_scorer(query, perspective, agent_memory, scorer):
    # Stage 1: Relevance Filter (unchanged)
    candidates = vector_search(
        query,
        agent_memory,
        limit=50,
        similarity_threshold=0.3
    )

    # Stage 2: Priority ranking with learned scorer
    task_context = TaskContext(perspective=perspective)

    for memory in candidates:
        # Extract features
        features = extract_features(memory, query, task_context, agent_memory.agent)
        feature_vector = torch.tensor([list(features.values())], dtype=torch.float32)

        # Predict with scorer
        with torch.no_grad():
            memory.final_score = scorer(feature_vector).item()

    # Re-rank and return top results
    ranked = sorted(candidates, key=lambda m: m.final_score, reverse=True)
    return ranked[:10]
```

**Learning Process**:

Update the scorer using task success/failure as feedback.

```python
def train_scorer(scorer, optimizer, task_result, used_memories, unused_candidates):
    """
    Train scorer based on task results

    Positive: Information actually used when task succeeded → high score is correct
    Negative: Information that became candidate but wasn't used → low score is correct
    """
    scorer.train()

    # Create positive labels
    positive_features = [extract_features(m, ...) for m in used_memories]
    positive_labels = torch.ones(len(used_memories), 1)

    # Create negative labels (sample subset)
    negative_samples = random.sample(unused_candidates, min(len(unused_candidates), len(used_memories) * 2))
    negative_features = [extract_features(m, ...) for m in negative_samples]
    negative_labels = torch.zeros(len(negative_samples), 1)

    # Weight by task result
    if task_result == "success":
        weight = 1.0  # Normal weight on success
    else:
        weight = 0.5  # Weaker learning on failure (might be causes other than info selection)

    # Loss calculation and update
    all_features = torch.tensor(positive_features + negative_features, dtype=torch.float32)
    all_labels = torch.cat([positive_labels, negative_labels])

    predictions = scorer(all_features)
    loss = nn.BCELoss()(predictions, all_labels) * weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Learning Stabilization**:

```python
class ScorerWithFallback:
    """
    Fall back to linear scoring in early learning stage
    Use learned scorer after sufficient data accumulated
    """
    def __init__(self, scorer, min_training_samples=100):
        self.scorer = scorer
        self.min_training_samples = min_training_samples
        self.training_samples = 0

    def score(self, features):
        if self.training_samples < self.min_training_samples:
            # Linear scoring during early learning (traditional method)
            return (
                features['similarity'] * 0.40 +
                features['strength'] * 0.40 +
                features['recency'] * 0.20
            )
        else:
            # Use neural net after sufficient learning
            feature_vector = torch.tensor([list(features.values())], dtype=torch.float32)
            with torch.no_grad():
                return self.scorer(feature_vector).item()
```

**Design Summary**:

| Element | Design | Reason |
|---------|--------|--------|
| Features | 9 dimensions, fixed | Values computed from vector search and metadata |
| Network | 2-layer MLP | Small input dimension, prevent overfitting |
| Parameters | ~180 | Learnable with hundreds of tasks |
| Learning signal | Task success/failure | Same "worked well when used" feedback as humans |
| Fallback | Linear score | Stability during early learning |

With this design, scoring is automatically optimized during operation. As an agent's experience accumulates, the optimal scoring function for that agent is learned.

#### Search Miss Prevention: Query Expansion

Vector search has risk of misses due to expression differences:

```
Query: "Part delay response"
Memory: "Supply chain issues caused Supplier Y 2-week delay"

→ Semantically related but embedding similarity might be low
→ This is NOT a problem to solve with access_count
→ Handle with query expansion
```

```python
def expand_query(original_query, perspective):
    """Expand query to prevent search misses"""

    # Add perspective-related keywords
    perspective_keywords = {
        "Procurement": ["supplier", "delivery", "cost", "order", "delay"],
        "Quality": ["defect", "inspection", "standard", "complaint", "yield"],
        "Customer": ["satisfaction", "response", "trust", "complaint", "request"],
    }

    expanded = original_query
    if perspective in perspective_keywords:
        expanded += " " + " ".join(perspective_keywords[perspective])

    return expanded


def search_with_expansion(query, perspective, agent_memory):
    """Comprehensive search with query expansion"""

    # Search with original query
    results1 = vector_search(query, agent_memory, limit=30, similarity_threshold=0.3)

    # Search with expanded query
    expanded = expand_query(query, perspective)
    results2 = vector_search(expanded, agent_memory, limit=30, similarity_threshold=0.3)

    # Merge and deduplicate
    all_results = deduplicate(results1 + results2)

    # Re-rank with score synthesis
    return rerank_with_strength(all_results, perspective)
```

#### Information to Always Reference: Principle Tags

Some information should always be referenced regardless of task content:

```
Example:
"Never compromise on quality" as a management policy
→ Should be kept in mind for any task
→ But low similarity to "budget approval" query
```

Handle this with **separate tag management**:

```json
{
  "id": "mem_001",
  "content": "Never compromise on quality",
  "tags": ["principle", "Quality"],
  "strength": 10.0,
  "source": "policy"
}
```

```python
def retrieve_with_principles(query, perspective, agent_memory):
    """Search that always includes fundamental principles"""

    # Normal search
    task_memories = search_with_expansion(query, perspective, agent_memory)

    # Always retrieve principles regardless of similarity
    principles = agent_memory.get_by_tag("principle")

    # Merge (always include principles, deduplicate)
    return deduplicate(principles + task_memories)
```

#### On Scalability

**Q: Will we need to get Top 1000 as memories grow?**

**A: No.** Reasons:

1. **Purpose of relevance filter**
   - Not "don't miss important information"
   - But "exclude irrelevant information"

2. **Scale difference**
   - Google search: From billions of pages
   - Agent memory: Thousands to tens of thousands
   - Top 50 provides sufficient coverage

3. **Low similarity = Irrelevant**
   - This is correct behavior
   - Even with high access_count, don't show if irrelevant

4. **Cases where misses truly matter**
   - Handle with query expansion
   - Principles managed separately with tags

#### The "Cliff" Problem and Countermeasures

Cutting search results at fixed count (e.g., Top 10) creates discontinuous treatment despite continuous scores:

```
Scores:
Top 10: 0.51  ← Enters context → Might be used → Might be strengthened
Top 11: 0.50  ← Doesn't enter → Not used → Not strengthened → Weakens through decay
```

This risks "rich get richer" inequality expansion.

**Mitigating factors**:

Even with cliff on single query, many queries average out to smoothing:

```
Query A: Information X is Top 8  → Enters
Query B: Information X is Top 15 → Doesn't enter
Query C: Information X is Top 3  → Enters
Query D: Information X is Top 12 → Doesn't enter

→ Long-term, "truly useful information" ranks high across various queries
→ Even if Top 11 once, can recover in other queries
```

Human memory works the same: Not recalled in one context, but recalled in another.

**Countermeasure options**:

Countermeasure proposals if problems are observed:

```python
# Option 1: Score threshold method (cut by score, not count)
def select_by_threshold(ranked_memories, score_threshold=0.4, max_count=20):
    selected = []
    for memory in ranked_memories:
        if memory.final_score >= score_threshold and len(selected) < max_count:
            selected.append(memory)
    return selected
    # → Results might be 5 or 15 items


# Option 2: Probabilistic selection (gentle slope, not cliff)
import random

def select_probabilistic(ranked_memories, base=10, extended=20):
    selected = ranked_memories[:base]  # Top 10 guaranteed

    for memory in ranked_memories[base:extended]:
        # Add with probability based on score
        if random.random() < memory.final_score:
            selected.append(memory)

    return selected
```

**Recommended approach**:

1. Start with fixed count (Top 20, generous) for initial operation
2. Monitor for bias
3. If problems observed, migrate to Option 1 or 2

```python
# Bias monitoring metrics
def detect_never_used_memories(agent_memory):
    """Detect information that becomes candidate but never used"""
    warnings = []
    for memory in agent_memory.all():
        if memory.candidate_count > 50 and memory.access_count == 0:
            # Became candidate 50 times but never used
            warnings.append(memory)
    return warnings
```

### 3.6 Strength Management and Memory Consolidation

#### Clarification of Strength's Role

Strength has two distinct roles:

| Role | Description | Timing |
|------|-------------|--------|
| **Existence** | Archive threshold (retire below threshold) | Sleep phase |
| **Search ranking** | Auxiliary influence on ranking | Search time (but similarity is primary) |

**Important design decision**: Search ranking is primarily determined by similarity (context matching), with strength playing only an auxiliary role. This allows differentiation by similarity even as consolidated memories increase.

```
Essential role of strength:
× Whether to appear high in search results
○ Whether to continue existing (resistance to forgetting)
```

#### Memory Consolidation

Like human memory, repeatedly used memories become "consolidated" and decay more slowly.

**Conversion from episodic to semantic memory**:

```
Episodic memory: "In January 2024, Tanaka was troubled by the Supplier Y fire"
     ↓ Repeated use + sleep consolidation
Semantic memory: "Supplier Y has single-site risk" (context detached)
```

**Consolidation Level Management**:

```python
class Memory:
    strength: float           # Current strength
    access_count: int         # Usage count
    consolidation_level: int  # Consolidation level (0-5)
    status: str               # "active" | "archived"

# Consolidation level thresholds
CONSOLIDATION_THRESHOLDS = [0, 5, 15, 30, 60, 100]  # access_count thresholds

def update_consolidation(memory):
    """Update consolidation level based on usage count"""
    for level, threshold in enumerate(CONSOLIDATION_THRESHOLDS):
        if memory.access_count >= threshold:
            memory.consolidation_level = level
```

**Relationship between Consolidation Level and Decay Rate**:

Decay rates are calculated by dividing "target daily decay" by "expected task count." For example, if targeting 5%/day decay for unconsolidated memories with 10 tasks/day, per-task decay rate is `0.95^(1/10) ≈ 0.995`.

| Consolidation Level | access_count | Decay Rate/Task | Daily Decay at 10 tasks/day |
|---------------------|--------------|-----------------|----------------------------|
| 0 | 0-4 | 0.995 | ~5%/day |
| 1 | 5-14 | 0.997 | ~3%/day |
| 2 | 15-29 | 0.998 | ~2%/day |
| 3 | 30-59 | 0.999 | ~1%/day |
| 4 | 60-99 | 0.9995 | ~0.5%/day |
| 5 | 100+ | 0.9998 | ~0.2%/day |

```python
# Core design: Distribute target daily decay across expected tasks
EXPECTED_TASKS_PER_DAY = 10  # Adjust based on operations

DAILY_DECAY_TARGETS = {
    0: 0.95,   # Unconsolidated: target 5%/day decay
    1: 0.97,
    2: 0.98,
    3: 0.99,
    4: 0.995,
    5: 0.998,  # Fully consolidated: almost no decay
}

# Calculate per-task decay rates
DECAY_RATES = {
    level: target ** (1 / EXPECTED_TASKS_PER_DAY)
    for level, target in DAILY_DECAY_TARGETS.items()
}

def get_decay_rate(memory):
    return DECAY_RATES[memory.consolidation_level]
```

**Parameter Tuning Guidelines**:
- Adjust `EXPECTED_TASKS_PER_DAY` to match actual task frequency
- Higher task frequency environments: increase value (smaller decay per task)
- Lower task frequency environments: decrease value (larger decay per task)

#### Two-Stage Reinforcement Process (Separating Reference and Use)

**Problem**: Even if search returns 10 items, LLM typically uses only about 2. Reinforcing all items strengthens noise.

**Solution**: Separate "became search candidate" from "actually used."

```python
def retrieve_and_track(query, perspective, agent_memory):
    """Step 1: At search time (don't reinforce yet)"""
    results = search_with_expansion(query, perspective, agent_memory)

    # Only record being referenced as candidate (light count)
    for memory in results:
        memory.candidate_count += 1

    return results


def finalize_task(task_context, agent_memory):
    """Step 2: At task completion (reinforce after confirming use)"""

    # Analyze LLM output to identify actually used information
    used_memories = identify_used_memories(
        task_context.llm_output,
        task_context.retrieved_memories
    )

    for memory in task_context.retrieved_memories:
        if memory in used_memories:
            # Actually used → Proper reinforcement
            memory.access_count += 1
            memory.strength += 0.1
            memory.last_access = now()

            # Perspective-specific reinforcement
            if task_context.perspective:
                perspective = task_context.perspective
                memory.strength_by_perspective[perspective] += 0.15
        # else: Referenced but not used → Do nothing (candidate_count already increased)
```

#### Methods for Determining "Was Used"

```python
def identify_used_memories(llm_output, retrieved_memories):
    """Determine which memories were reflected in LLM output"""
    used = []

    for memory in retrieved_memories:
        # Method 1: Keyword matching (simple, low cost)
        if any(tag in llm_output for tag in memory.tags):
            used.append(memory)
            continue

        # Method 2: Content similarity (moderate accuracy)
        similarity = compute_similarity(memory.content, llm_output)
        if similarity > 0.3:
            used.append(memory)
            continue

    return used


def identify_used_memories_by_llm(llm_output, retrieved_memories, task):
    """Method 3: Ask LLM (most accurate but high cost, for important tasks)"""

    prompt = f"""
You generated the following response to a task.

Task: {task.summary}
Response: {llm_output}

List of referenced information:
{format_memories_with_index(retrieved_memories)}

Answer with comma-separated numbers of information that actually influenced the response.
Do not include information that did not influence it.
"""

    response = lightweight_llm.complete(prompt)  # Haiku, etc.
    used_indices = parse_indices(response)

    return [retrieved_memories[i] for i in used_indices]
```

#### Impact-Based Reinforcement

```python
def update_impact(memory, context):
    impact = 0

    # Positive feedback from user
    if context.user_feedback == "helpful":
        impact += 2.0

    # Contributed to task success
    if context.task_result == "success" and memory in context.used_memories:
        impact += 1.5

    # Prevented an error
    if memory.prevented_error:
        impact += 2.0

    memory.impact_score += impact
    memory.strength += impact * 0.2
```

### 3.7 Task Execution Flow

```
1. Task received (summary + pointer to details)

2. Search related information from external memory
   - Generate query based on summary
   - Multiple searches by role perspectives
   - Rank with score synthesis considering strength
   - Only increment candidate_count at this point

3. Selectively fetch needed details
   - Judge from summary and external memory info
   - Don't read all details (context conservation)

4. Task execution (LLM processing)

5. Identify used information and update strength
   - Analyze LLM output to identify actually used information
   - Only used info: access_count++, strength += 0.1
   - Also update relevant perspective's strength_by_perspective
   - Unused information is not reinforced

6. Extract and save new learnings
   - Ask LLM about perspective-specific learnings
   - Create new entry in external memory
   - Add links to related existing memories

7. Sleep phase (always executed on task completion)
   - Update consolidation levels
   - Apply decay
   - Archive below threshold
   - Forced pruning if over capacity

8. Session end (context clear)
```

**Important Design Decision: Sleep on Every Task**

Agents execute the sleep phase on every task completion. Reasons:

1. **Context Separation**
   - Conversation continuity with users is handled by the orchestrator
   - Agents receive clearly interpreted tasks
   - Agents don't need to remember conversation context

2. **Persistence Guarantee**
   - Learnings are immediately saved to external memory
   - Compaction problem is completely eliminated
   - No information loss between tasks

3. **Consistent Decay**
   - Decay is applied per task
   - Adapts to increased task volume from computer performance improvements
   - Flexible adjustment through decay rate parameters

#### Learning Extraction Prompt Example

```
You are a procurement agent.
From this task experience, extract one sentence of learning for each perspective:
- Impact on cost
- Impact on delivery
- Relationship with supplier
- Quality risk
- Future alternatives
Omit perspectives that don't apply.
```

### 3.8 Sleep Phase (For Specialized Agents)

#### Why Sleep is Needed

**Problem with real-time decay**:

```
09:00:00 Decay processing starts
09:00:01 Memories 1-100 decayed
09:00:02 Task arrives, search executes
         → Compare Memory 50 (decayed) with Memory 150 (not decayed)
         → Memory 150 appears "unfairly" high
         → Judgment inconsistency occurs
```

Tasks arriving during decay processing cause inconsistent memory state, breaking judgment consistency.

**Problem with backup approach**:

```
09:00 Create backup
09:05 "Major complaint" on production → Related memories greatly reinforced
09:10 Decay complete, switch to backup
      → Major complaint reinforcement is gone
      → Bad judgment made
```

#### Sleep Phase Implementation

The sleep phase executes three processes in sequence:

1. **Consolidation-based decay** - Well-used memories decay less
2. **Threshold-based archiving** - Retire weakened memories
3. **Capacity-based forced pruning** - Adjust when over limit

```python
def sleep_phase(agent_memory):
    """
    Execute on each task completion

    Design rationale:
    - Orchestrator maintains conversation context
    - Agents handle specialized judgment only
    - Per-task sleep ensures immediate persistence of learnings
    """

    # 1. Stop task acceptance (this agent instance will terminate)
    agent_memory.accepting_tasks = False

    # 2. Update consolidation levels
    for memory in agent_memory.active():
        update_consolidation(memory)

    # 3. Apply consolidation-based decay
    for memory in agent_memory.active():
        decay_rate = get_decay_rate(memory)

        memory.strength *= decay_rate
        for perspective in memory.strength_by_perspective:
            memory.strength_by_perspective[perspective] *= decay_rate

        # Note: With per-task sleep, "recent access" offset is unnecessary
        # Memories used just now already have increased strength,
        # so net effect of reinforcement minus decay naturally strengthens them

    # 4. Archive below threshold (logical forgetting)
    for memory in agent_memory.active():
        if memory.strength < 0.1 and all(
            s < 0.1 for s in memory.strength_by_perspective.values()
        ):
            memory.status = "archived"

    # 5. Capacity-based forced pruning
    prune_if_over_capacity(agent_memory)

    # 6. Resume task acceptance
    agent_memory.accepting_tasks = True
```

#### Capacity Management and Forced Pruning

**Why Capacity Limits are Needed**:

```
Problem: Consolidated memories keep growing during long-term operation
→ Search becomes slow
→ Too many similar memories make differentiation difficult
→ Unlike human brain, no physical constraints so growth is unlimited

Solution: Set explicit capacity limits
```

**Capacity Calculation Method**:

Instead of simple count, calculate capacity using "weight" based on consolidation level. More consolidated memories "take more space."

```python
class MemoryCapacity:
    MAX_TOTAL_WEIGHT = 10000  # Total weight limit for active memories

    # Weight by consolidation level (more consolidated = heavier)
    CONSOLIDATION_WEIGHTS = {
        0: 1.0,   # Unconsolidated: light
        1: 2.0,
        2: 4.0,
        3: 8.0,
        4: 16.0,
        5: 32.0,  # Fully consolidated: heavy
    }

    def calculate_weight(self, memory):
        """Calculate memory's 'weight'"""
        return self.CONSOLIDATION_WEIGHTS[memory.consolidation_level]

    def get_total_weight(self, agent_memory):
        """Calculate total weight of active memories"""
        return sum(
            self.calculate_weight(m)
            for m in agent_memory.active()
        )
```

**Forced Pruning Implementation**:

```python
def prune_if_over_capacity(agent_memory):
    """Forced pruning when over capacity"""
    capacity = MemoryCapacity()
    total = capacity.get_total_weight(agent_memory)

    if total <= capacity.MAX_TOTAL_WEIGHT:
        return  # Do nothing if within capacity

    # Sort pruning candidates
    # Priority: lower consolidation first → older last access first
    candidates = sorted(
        agent_memory.active(),
        key=lambda m: (m.consolidation_level, -m.days_since_last_access)
    )

    # Archive until within capacity
    for memory in candidates:
        if total <= capacity.MAX_TOTAL_WEIGHT:
            break
        memory.status = "archived"
        total -= capacity.calculate_weight(memory)
```

**Pruning Priority**:

```
1. Prioritize archiving less consolidated memories
2. Among same consolidation level, prioritize older last access
3. Even level 5 (fully consolidated) can be pruned for capacity

→ No "never forget" memories exist
→ But they're harder to prune (remain until last)
```

This follows the same principle as synaptic pruning in the human brain: to create new memories, you need to reuse the "space" of old unused memories.

#### Archive and Re-activation

**Archive is not deletion**:

Human memories may not be "erased" but just "inaccessible." With the right cue (hint), they can be recalled.

```python
def search(query, agent_memory):
    """Normal search: active only"""
    candidates = vector_search(query, agent_memory, status="active")
    return candidates

def deep_recall(query, agent_memory):
    """Deep recall: include archived"""
    # Explicit recall like "that time... what was it?"
    candidates = vector_search(query, agent_memory, status="all")

    # If recalled, return to active (re-activation)
    for memory in candidates:
        if memory.status == "archived":
            memory.status = "active"
            memory.strength = 0.5  # Initial strength on re-activation
            memory.consolidation_level = max(0, memory.consolidation_level - 2)

    return candidates
```

#### Design Summary

| Process | Timing | Target | Effect |
|---------|--------|--------|--------|
| Reinforcement | Task execution | Used memories | strength++, access_count++ |
| Consolidation update | Sleep phase | All active memories | Update consolidation_level |
| Decay | Sleep phase | All active memories | Decrease strength based on consolidation |
| Archive | Sleep phase | Below strength threshold | status→archived |
| Forced pruning | Sleep phase | When over capacity | Archive lowest consolidation first |
| Re-activation | On deep_recall | Archived memories | status→active |

#### Background Replay is Unnecessary

The main purpose of human sleep replay is "transfer from hippocampus to neocortex." Agent external memory is already a long-term storage location, so no transfer processing equivalent is needed.

What's needed:
1. Reinforcement through reference (during task execution, real-time after use confirmation)
2. Consolidation-based decay (on task completion, in sleep phase)
3. Capacity management and forced pruning (on task completion, in sleep phase)
4. Impact bonus (during task execution, real-time)

These four are sufficient. Simpler is correct.

#### Sleep Timing Design

```
[Traditional thinking]
Sleep = Daily batch processing
    ↓
Problem: As computer performance improves, task volume increases
         Large amounts of information accumulate within a day
         Decay can't keep up

[This Architecture]
Sleep = On every task completion
    ↓
Benefits:
- Constant decay regardless of task volume
- Immediate persistence of learnings
- Complete elimination of compaction problem
- Conversation context managed by orchestrator
```

### 3.9 Addressing Compaction Problem

#### Problem

When an agent runs for a long time, context window fills and compaction occurs. Information not yet written to external memory is permanently lost.

#### Solution: One Task, One Session

```
[Traditional thinking]
Agent = Long-running process
    ↓
Compaction problem occurs

[New thinking]
Agent = Start/stop per task
External memory = Persistent identity
    ↓
Compaction problem doesn't occur
```

By always writing learnings to external memory at task completion and ending the session, important information is persisted before compaction.

### 3.10 Input Processing Layer

#### Separation of Summary and Details

Design to prevent orchestrator from overflowing context with large inputs.

```python
class Task:
    summary: str          # Required, under 1000 tokens
    detail_refs: List[str] # Pointers to details (S3, DB, etc.)
```

Orchestrator judges from summary only; specialized agents view only needed parts of details.

#### Generating Summary When Absent

```
100 pages → Mechanical split into 10-page chunks
    ↓
Extract 1 sentence per 10 pages (parallelizable, lightweight LLM)
    ↓
Combine 10 sentences → Summary (~500 tokens)
```

#### Detecting and Negotiating Excessive Input

```python
class InputProcessor:
    def process(self, input):
        items = self.detect_multiple_items(input)

        if len(items) > THRESHOLD:  # e.g., 20 items
            return {
                "type": "negotiation_needed",
                "message": f"There are {len(items)} items.",
                "options": [
                    "Please specify the 10 highest priority",
                    "Process all (will take time)",
                    "Divide by category and process sequentially"
                ]
            }

        return {"type": "ok", "items": items}
```

---

## Chapter 4: Formation of Expertise

### 4.1 Education Process: Seeds of Expertise

A new agent's external memory starts empty. A process equivalent to "education" in humans is needed.

#### Textbook + Test Method

Form expertise through the same process as human education:

```
Textbook (specialized book) + Quizzes
    ↓
Read in chunks
    ↓
Execute tests as tasks
    ↓
Sleep (decay processing)
    ↓
Repeat
```

#### Implementation

```python
class EducationProcess:
    def __init__(self, agent, textbook_path, perspective):
        self.agent = agent
        self.textbook = load_textbook(textbook_path)
        self.perspective = perspective

    def run(self):
        chapters = self.textbook.chapters

        for chapter in chapters:
            # Step 1: Read (chunk and process sequentially)
            self.read_chapter(chapter)

            # Step 2: Sleep (decay processing)
            self.agent.sleep()

            # Step 3: Test
            self.take_test(chapter.test_questions)

            # Step 4: Sleep
            self.agent.sleep()

    def read_chapter(self, chapter):
        for section in chapter.sections:
            # Save read content to external memory
            memory = {
                "content": section.text,
                "learnings": self.extract_learnings(section, self.perspective),
                "tags": extract_tags(section),
                "strength": 0.5,  # Initial strength is low (not "used" yet)
                "source": "education"
            }
            self.agent.external_memory.save(memory)

    def take_test(self, questions):
        for question in questions:
            # Execute test as task
            task = Task(
                summary=question.text,
                detail_refs=[]
            )

            result = self.agent.execute(task)

            # Check answer
            if self.check_answer(result, question.answer):
                # Correct: Used information automatically reinforced
                # (already done in normal task completion processing)
                pass
            else:
                # Incorrect: Explicitly reinforce related correct information (review)
                self.reinforce_correct_knowledge(question)

    def reinforce_correct_knowledge(self, question):
        """Review processing when incorrect"""
        # Search memories related to correct answer
        related = self.agent.external_memory.search(question.answer)

        for memory in related[:5]:
            # Reinforce as "review"
            memory.strength += 0.2
            memory.access_count += 1
            memory.last_access = now()
```

#### Process Flow and Effects

```
┌────────────────────────────────────────────────────────────┐
│  Education Phase                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Chapter 1                                                 │
│  ├── Read → Save to external memory (strength: 0.5)       │
│  ├── Sleep → Decay (just-read info weakens)               │
│  ├── Test → Info used for correct answer reinforced (0.6+)│
│  └── Sleep → Decay (unused info weakens further)          │
│                                                            │
│  Chapter 2                                                 │
│  ├── Read → Save new content                              │
│  ├── Sleep                                                 │
│  ├── Test → Might use Ch1 knowledge → Re-reinforced       │
│  └── Sleep                                                 │
│                                                            │
│  ... Repeat ...                                            │
│                                                            │
│  Result:                                                   │
│  ├── Knowledge repeatedly used in tests → High strength   │
│  ├── Knowledge only read, not used → Low strength         │
│  └── "Important knowledge" naturally remains              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Textbook Structure Example

```yaml
textbook:
  title: "Procurement Basics"
  perspective: "Procurement"

  chapters:
    - title: "Supplier Evaluation"
      sections:
        - title: "Quality Evaluation Criteria"
          text: "When evaluating supplier quality..."
        - title: "Cost Analysis"
          text: "From a Total Cost of Ownership (TCO) perspective..."
      test_questions:
        - question: "Given the following information about Suppliers A and B, which should you choose? Explain your reasoning."
          answer_keywords: ["TCO", "quality", "risk diversification"]

    - title: "Contract Negotiation"
      sections:
        - title: "Price Negotiation Basics"
          text: "..."
      test_questions:
        - question: "..."
```

#### Spaced Repetition

"Spaced repetition" effective in human learning can also be implemented:

```python
def spaced_repetition(agent, test_questions):
    """Spaced repetition schedule"""
    schedule = [1, 3, 7, 14, 30]  # Days

    for days in schedule:
        wait(days)

        for question in test_questions:
            result = agent.execute(Task(summary=question.text))

            if check_answer(result, question.answer):
                # Correct: Used information reinforced
                # Can lengthen next review interval
                pass
            else:
                # Incorrect: Review, shorten next interval
                reinforce_correct_knowledge(question)

        agent.sleep()
```

### 4.2 Why Personality Emerges

**Same LLM (identical parameters) develops personality through different external memory**:

```
Agent A: Many procurement tasks
    → Procurement-related memories reinforced
    → Procurement perspective strength_by_perspective is high
    → Procurement-related items rank higher in search
    → Thinks from procurement perspective even for new tasks
    → Personality of "Agent A strong in procurement"

Agent B: Many quality tasks
    → Quality-related memories reinforced
    → Personality of "Quality-focused Agent B"
```

### 4.3 Formation of Consistency

```
Past decision: "Prioritized quality over delivery" → Success
    ↓
Recorded in external memory, impact_score increases
    ↓
Referenced with high strength in similar situations
    ↓
Makes quality-priority decision again
    ↓
Consistency of "quality-focused" personality
```

### 4.4 Differentiation of Perspectives

```
[Current Multi-agent]
Agent A: System prompt + current task info
Agent B: System prompt + current task info
→ Everyone same perspective, converging to same conclusion

[This Architecture]
Agent A: Common prompt + task + procurement-specialized external memory (strength distribution skewed toward procurement)
Agent B: Common prompt + task + quality-specialized external memory (strength distribution skewed toward quality)
→ Different perspectives, different proposals
→ Discussion emerges
→ Organizational wisdom
```

---

## Chapter 5: Technical Considerations

### 5.1 Technically Ambiguous Points

#### Impact Score Definition

- Criteria for determining "contributed to task success"
- Evaluation of indirect influence

**Response**: Start simple with "used information +1" and refine through operation.

#### Parameter Tuning

- Decay rate (0.95? 0.99?)
- Threshold (0.1? 0.05?)
- Reinforcement amount (+0.1? +0.2?)
- Score synthesis weights (similarity 0.4, strength 0.4, recency 0.2)

**Response**: No correct answer; tune through operation based on domain and task frequency.

#### Cliff Problem

Cutting search results at fixed count (Top N) creates discontinuous treatment for information near the boundary. Score difference between Top 10 and Top 11 may be tiny, but treatment differs greatly.

**Mitigating factors**: Averaged across many queries, smoothing occurs. Truly useful information ranks high across various queries, so one Top 11 is recovered in other queries.

**Response**:
- Start with fixed count (generous Top 20)
- Monitor bias through ratio of `candidate_count` to `access_count`
- If problems observed, migrate to score threshold or probabilistic selection

### 5.2 Technical Difficulties

**No major difficulties**. All necessary components achievable with existing technology.

| Component | Existing Technology |
|-----------|---------------------|
| Vector DB | Pinecone, Qdrant, Chroma, etc. |
| Metadata management | Standard RDB |
| LLM API | Claude, GPT, etc. |
| Periodic batch | cron, Cloud Scheduler, etc. |
| Orchestration | Python, LangGraph, etc. |

No innovative new technology required. Achievable through **combination of existing components**.

### 5.3 Implementation Effort Estimate

| Phase | Content |
|-------|---------|
| PoC | Single agent, basic reinforcement/decay, verification |
| Production Level | Multiple agents, parameter tuning, error handling, monitoring |
| Education Process | Textbook structure design, test creation, review scheduling |

### 5.4 Remaining Constraints

#### Limits of "Unreasonable Requests"

Cases like throwing 102 deep design questions at once:
- Detection and warning possible in input processing layer
- Split processing options can be offered
- However, pre-estimating "depth" is difficult
- Proper request design by the requester is necessary

**This is the same constraint even with humans**. Perfect systems can't be built, but "saying it's impossible" or "proposing splits" is better than "freezing silently."

---

## Chapter 6: Comparison with Related Research

Multiple related research efforts exist for this architecture. This chapter clarifies the overview of these studies and the differences from this architecture.

### 6.1 MemoryBank (AAAI 2024)

**Overview**: [MemoryBank](https://arxiv.org/abs/2305.10250) is a mechanism for providing long-term memory to LLMs. It adopts a decay mechanism based on Ebbinghaus forgetting curve theory, forgetting and strengthening memories according to time elapsed.

**Key Features**:
- Three pillars: memory storage, memory retriever, memory updater
- Time-based decay using Ebbinghaus forgetting curve (R = e^(-t/S))
- Demonstrated with SiliconFriend AI companion

**Differences from This Architecture**:

| Aspect | MemoryBank | This Architecture |
|--------|------------|-------------------|
| Decay basis | Time only | Time + usage frequency + impact |
| Reinforcement judgment | Reinforce when referenced | **Reinforce only when actually used** (2-stage) |
| Perspective management | None | **Perspective-specific strength management** |
| Sleep phase | None | **Batch processing for consistency** |

MemoryBank is a simple model of "forget as time passes." This architecture more faithfully mimics the brain's LTP principle of "unused things are forgotten, used things are strengthened."

### 6.2 LightMem (2025)

**Overview**: [LightMem](https://arxiv.org/abs/2510.18866) is a 3-stage memory system based on the Atkinson-Shiffrin memory model. Focuses on efficiency, reducing token usage by up to 117x.

**Key Features**:
- Sensory Memory: Filtering through lightweight compression
- Short-term Memory: Topic-based organization and summarization
- Long-term Memory: Offline integration through sleep-time update

**Differences from This Architecture**:

| Aspect | LightMem | This Architecture |
|--------|----------|-------------------|
| Sleep phase | Yes (integration) | Yes (decay processing) |
| Strength management | None | **Usage-based strength management** |
| Expertise formation | None | **Education process** |
| Personality emergence | None | **Personality through perspective-specific strength** |

LightMem focuses on efficiency and lacks a mechanism for managing memory "importance" by usage frequency.

### 6.3 Language Models Need Sleep (OpenReview 2025)

**Overview**: [Language Models Need Sleep](https://openreview.net/forum?id=iiZy6xyVVE) introduces a biological "sleep" paradigm to LLMs. Achieves continual learning through two stages: Memory Consolidation and Dreaming.

**Key Features**:
- Memory Consolidation: Parameter expansion through RL-based Knowledge Seeding
- Dreaming: Self-improvement phase with synthetic data
- Improved resistance to catastrophic forgetting

**Differences from This Architecture**:

| Aspect | LM Need Sleep | This Architecture |
|--------|---------------|-------------------|
| Parameter update | **Yes** | No |
| Implementation difficulty | High (model modification required) | Medium (existing technology) |
| Depth of expertise | Deep (internalized) | Shallow (external reference) |
| Production status | Research stage | **Immediately implementable** |

This research enables true "learning" through parameter updates but has high implementation barriers. This architecture aims for practical effects even with immutable parameters.

### 6.4 MemGPT / Letta

**Overview**: [MemGPT](https://arxiv.org/abs/2310.08560) is an LLM memory system inspired by OS virtual memory management. Now evolved as [Letta](https://docs.letta.com/).

**Key Features**:
- 2-tier memory hierarchy: In-context (RAM equivalent) and Out-of-context (disk equivalent)
- Self-editing memory: LLM itself controls memory movement
- Heartbeat mechanism: Multi-step reasoning support
- Separation of Archival Memory and Recall Memory

**Differences from This Architecture**:

| Aspect | MemGPT/Letta | This Architecture |
|--------|--------------|-------------------|
| Forgetting mechanism | **None** | Decay + sleep phase |
| Strength management | None | **Usage-based strength management** |
| Expertise formation | None | **Education process** |
| Memory structure | Hierarchical (RAM/disk) | Flat (managed by strength) |

MemGPT focuses on creating the illusion of "infinite memory" and lacks a judgment mechanism for "what to forget and what to remember."

### 6.5 A-MEM (NeurIPS 2025)

**Overview**: [A-MEM](https://arxiv.org/abs/2502.12110) is a self-organizing memory system based on the Zettelkasten method. Features dynamic index creation and link generation.

**Key Features**:
- Zettelkasten principle: Interconnected knowledge networks
- Dynamic index generation: Automatic linking when adding new memories
- Memory evolution: New memories update existing memories

**Differences from This Architecture**:

| Aspect | A-MEM | This Architecture |
|--------|-------|-------------------|
| Forgetting mechanism | **None** | Decay + sleep phase |
| Strength management | None | **Usage-based strength management** |
| Organization method | Link-based | Strength-based |
| Search method | Graph traversal | Vector + strength score synthesis |

A-MEM focuses on memory "organization" and doesn't handle "importance changes" over time or by usage frequency.

### 6.6 HippoRAG (NeurIPS 2024)

**Overview**: [HippoRAG](https://arxiv.org/abs/2405.14831) is a RAG framework based on hippocampal indexing theory. Features multi-hop reasoning combining Knowledge Graph and Personalized PageRank.

**Key Features**:
- Knowledge Graph generation by LLM
- Subgraph exploration through Personalized PageRank
- Up to 20% performance improvement in multi-hop Q&A

**Differences from This Architecture**:

| Aspect | HippoRAG | This Architecture |
|--------|----------|-------------------|
| Main purpose | Search accuracy improvement | Memory persistence & personality formation |
| Forgetting mechanism | **None** | Decay + sleep phase |
| Strength management | None | **Usage-based strength management** |
| Graph structure | Required | Not required (vector DB sufficient) |

HippoRAG focuses on optimizing "search" and doesn't handle "memory management" (what to forget, what to strengthen).

### 6.7 Google Titan (2024)

**Overview**: [Titans](https://arxiv.org/abs/2501.00663) introduces a Neural Memory Module that learns at test time. Supports context windows larger than 2M.

**Key Features**:
- Separation of short-term memory (Attention) and long-term memory (Neural Memory)
- Surprise-based learning: Using gradients as "surprise" signal for memory decisions
- Adaptive forgetting: Decay of old memories through weight decay
- Three variants: MAC, MAG, MAL

**Differences from This Architecture**:

| Aspect | Titan | This Architecture |
|--------|-------|-------------------|
| Parameter update | **Yes** (Neural Memory) | No |
| Production status | Research stage | **Immediately implementable with existing tech** |
| Depth of expertise | Deep (internalized) | Shallow (external reference) |
| Implementation complexity | High | Medium |
| Forgetting judgment | Surprise metrics | **Usage frequency + impact** |

Titan is an ideal solution but takes time to become production-ready. This architecture is positioned as a "practical solution until Titan becomes production-ready."

### 6.8 Continual Learning Research (EWC, Synaptic Intelligence)

**Overview**: [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796) and [Synaptic Intelligence](https://www.pnas.org/doi/10.1073/pnas.1611835114) are regularization methods to prevent catastrophic forgetting in neural networks.

**Key Features**:
- Penalize changes to important parameters
- Based on biological principles of synaptic plasticity
- Reuse shared structures across tasks

**Differences from This Architecture**:

| Aspect | EWC/SI | This Architecture |
|--------|--------|-------------------|
| Target | Model parameters | **External memory** |
| Purpose | Prevent forgetting during fine-tuning | Memory management during task execution |
| Parameter update | Yes | **No** |

These methods handle problems during model training, while this architecture handles external memory management during inference.

### 6.9 Uniqueness of This Architecture

From comparison with the related research above, this architecture's uniqueness lies in:

1. **Two-stage reinforcement process (candidate vs. use)**
   - Don't reinforce just for becoming a search candidate
   - Only reinforce information actually used by LLM
   - Prevent noise reinforcement, retain truly useful information
   - **Novel contribution not found in other research**

2. **Perspective-specific strength management**
   - Same information has different importance by perspective
   - Realize "personality" of specialized agents through external memory
   - **Novel contribution not found in other research**

3. **Practical effects with immutable parameters**
   - Implementable with existing technology only
   - Realistic solution usable starting today
   - Bridge to future technologies like Titan

4. **Expertise formation through education process**
   - Same "textbook + test" process as humans
   - Repetitive learning with sleep phases in between
   - Natural establishment of specialized knowledge

### 6.10 Related Research Summary Table

| Research | Forgetting | Use Reinforcement | Perspective | Param Immutable | Sleep | Easy to Implement |
|----------|:----------:|:-----------------:|:-----------:|:---------------:|:-----:|:-----------------:|
| MemoryBank | ○ | × | × | ○ | × | ○ |
| LightMem | △ | × | × | ○ | ○ | ○ |
| LM Need Sleep | ○ | △ | × | × | ○ | × |
| MemGPT/Letta | × | × | × | ○ | × | ○ |
| A-MEM | × | × | × | ○ | × | ○ |
| HippoRAG | × | × | × | ○ | × | ○ |
| Google Titan | ○ | ○ | × | × | × | × |
| **This Architecture** | **○** | **○** | **○** | **○** | **○** | **○** |

---

## Chapter 7: Summary

### 7.1 Core Design

1. **Agent body is stateless, external memory carries identity**

2. **Two-stage search structure**: Stage 1 (relevance filter) excludes irrelevant information, Stage 2 (priority ranking) factors in strength and recency. Low similarity information doesn't appear even with high access_count (this is correct behavior)

3. **Two-stage reinforcement process**: Don't reinforce just for becoming a search candidate; only reinforce actually used information (prevent noise reinforcement)

4. **Periodic decay**: Batch processing in sleep phase (ensure consistency)

5. **Pre-defined perspectives**: Structure learnings with ~5 perspectives according to role

6. **Education process**: Repeat textbook + test with sleep in between to form seeds of expertise

7. **One task, one session**: Persist learnings before compaction

8. **Separation of summary and details**: Orchestrator judges from summary only

9. **Exception handling**: Prevent search misses with query expansion, manage principles separately with tags

### 7.2 Correspondence with Human Organizations

| Human Organization | This Architecture |
|-------------------|-------------------|
| Individual memory | Agent-dedicated external memory |
| Expertise | Strength distribution by perspective |
| Experience accumulation | Strength increase of used information |
| Forgetting | Decay processing |
| Sleep | Sleep phase (decay batch) |
| Education/training | Textbook + test process |
| Review | Spaced Repetition |
| Manager's summary check | Orchestrator's summary routing |
| Handoff of details to staff | Passing detail pointers |

### 7.3 Abandon Perfection

Human organizations also:
- Sometimes forget important experiences
- Think "should have noticed then" in hindsight
- Commonly have perspective gaps
- **Yet somehow manage**

Agents should similarly aim for "better than now" rather than perfection.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| External memory | Agent-dedicated persistent memory store (vector DB + metadata) |
| Strength | Importance score of memory. Increases with use, decays periodically. Determines existence (archive threshold) |
| access_count | Number of times actually used. Used for consolidation level updates |
| candidate_count | Number of times became search candidate but not used |
| Perspective | Judgment viewpoint according to agent role (e.g., Cost, Delivery) |
| Sleep phase | Period where task acceptance stops for batch decay, archive, and pruning |
| Impact score | Additional reinforcement from feedback or task success |
| Compaction | Information compression/deletion when context window fills |
| LTP (Long-Term Potentiation) | Brain phenomenon where synapse firing strengthens that synapse |
| Relevance filter | Search Stage 1. Exclude irrelevant information by vector similarity |
| Priority ranking | Search Stage 2. Rank within relevant candidates (similarity is primary) |
| Query expansion | Method to prevent vector search misses by adding perspective keywords |
| Principle tag | Special tag for information to always reference regardless of similarity |
| Learnable scorer | Neural network that predicts search scores from features |
| Cliff problem | Problem of discontinuous treatment for boundary information when cutting results at fixed count |
| Education process | Process to form seeds of expertise through textbook + test |
| Consolidation level | Level (0-5) indicating how well a memory is consolidated based on access_count |
| Decay rate | Rate at which strength decreases during sleep phase, varies by consolidation level |
| Archive | Logical forgetting. Excluded from search but not deleted |
| Re-activation | Returning archived memory to active status through explicit deep recall |
| Capacity limit | Upper limit on total weight of active memories. Prevents breakdown during long-term operation |
| Forced pruning | Process of archiving memories in order of lowest consolidation when over capacity |
| Orchestrator | Agent operating with the same mechanism as specialized agents. Only role and perspectives differ |
| Unified design | Design principle where orchestrator and specialized agents share the same mechanisms (external memory, strength management, decay, etc.) |
| Implicit feedback | User signals detected without explicit statement (e.g., using result as-is = success) |

## Appendix B: Related Research

### B.1 LLM Memory Systems

| Research | Published | Summary | Paper |
|----------|-----------|---------|-------|
| MemoryBank | AAAI 2024 | Time-decay memory based on Ebbinghaus forgetting curve | [arXiv:2305.10250](https://arxiv.org/abs/2305.10250) |
| LightMem | 2025 | Atkinson-Shiffrin 3-stage memory, sleep-time update | [arXiv:2510.18866](https://arxiv.org/abs/2510.18866) |
| Language Models Need Sleep | OpenReview 2025 | Continual learning through Memory Consolidation + Dreaming | [OpenReview](https://openreview.net/forum?id=iiZy6xyVVE) |
| MemGPT / Letta | NeurIPS 2023 | OS-style virtual memory management, self-editing memory | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) |
| A-MEM | NeurIPS 2025 | Self-organizing memory using Zettelkasten method | [arXiv:2502.12110](https://arxiv.org/abs/2502.12110) |
| HippoRAG | NeurIPS 2024 | Hippocampal indexing theory + PageRank | [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) |
| Google Titan | 2024 | Test-time learning Neural Memory, Surprise-based learning, adaptive decay | [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) |
| MemLong | 2024 | External memory retrieval for long text modeling, retrieval-frequency-based pruning | [arXiv:2408.16967](https://arxiv.org/abs/2408.16967) |
| H-MEM | 2025 | Hierarchical memory architecture, multi-level search with positional indexing | [arXiv:2507.22925](https://arxiv.org/abs/2507.22925) |
| Mem0 | 2025 | Production-ready agent long-term memory, graph-based memory representation | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) |

### B.2 Continual Learning & Catastrophic Forgetting

| Research | Published | Summary | Paper |
|----------|-----------|---------|-------|
| Elastic Weight Consolidation | PNAS 2017 | Regularization penalizing changes to important parameters | [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) |
| Synaptic Intelligence | ICML 2017 | Continual learning based on synaptic plasticity | [PNAS](https://www.pnas.org/doi/10.1073/pnas.1611835114) |

### B.3 Neuroscience Theories

| Research | Summary |
|----------|---------|
| Synaptic Homeostasis Hypothesis (SHY) | Theory that synapses globally downscale during sleep, relatively emphasizing important memories |
| Long-Term Potentiation (LTP) | Phenomenon where synapse firing strengthens that synapse. Neuroscientific basis of "Use it or lose it" |
| Ebbinghaus Forgetting Curve | Law that memory retention decays exponentially over time (R = e^(-t/S)) |
| Spaced Repetition | Learning technique promoting memory retention through spaced review |

### B.4 Spaced Repetition & Memory Strength Models

| Research | Published | Summary | Paper/Link |
|----------|-----------|---------|-------|
| FSRS | 2022- | ML-optimized spaced repetition scheduler based on DSR model | [GitHub](https://github.com/open-spaced-repetition/fsrs4anki) |
| DRL-SRS | Applied Sciences 2024 | Deep reinforcement learning for spaced repetition optimization, half-life regression model | [MDPI](https://www.mdpi.com/2076-3417/14/13/5591) |

### B.5 Survey Papers

| Research | Published | Summary | Paper |
|----------|-----------|---------|-------|
| From Human Memory to AI Memory | 2025 | Comprehensive survey systematizing the relationship between human memory and AI memory | [arXiv:2504.15965](https://arxiv.org/abs/2504.15965) |
| Memory in the Age of AI Agents | 2025 | 102-page unified framework for agent memory (Forms, Functions, Dynamics) | [arXiv:2512.13564](https://arxiv.org/abs/2512.13564) |

---

*This document was created based on discussions on January 11, 2025.*
