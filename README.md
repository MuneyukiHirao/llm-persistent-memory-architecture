# LLM Persistent Memory Architecture

[日本語版 / Japanese](README.ja.md)

> **"A 60,000-person organization works collaboratively despite individuals not sharing context windows. Why can't LLM agents do the same?"**

---

An architecture design to solve the fundamental limitation of LLM agents: "memory persistence". This proposal applies insights from neuroscience (LTP, memory consolidation during sleep) to software design, offering a practical solution implementable with existing technologies only.

---

## The Core Problem

### Current Limitations of LLM Agents

1. **Session Reset** - "Reborn" each time, no accumulation of experience
2. **Lost in the Middle Problem** - Even with 1M token context, only a portion is effectively usable
3. **Silent Compaction** - Information is lost without knowing what was discarded
4. **Lack of Meta-knowledge** - No organizational knowledge of "who knows what"

### The "Invisible Infrastructure" of Human Organizations

Human organizations have:

- Shared tacit knowledge and culture
- Persistent identity and relationships
- Asynchronous but persistent "organizational memory"
- A "something feels off" sensor that detects anomalies

## Core Insight

```
What's persistent is not the agent itself, but the external memory

Agent: Reborn each time (stateless)
External Memory: Persistent (stateful)
Identity: Carried by external memory
```

## Architecture Features

### 1. Strength Management Inspired by Brain's LTP

When information is referenced, its strength counter increments. Implementing "Use it or lose it" in software.

```python
# Two-stage reinforcement process
# Stage 1: Became search candidate → candidate_count++ (light count)
# Stage 2: Actually used → access_count++, strength += 0.1 (full reinforcement)
```

### 2. Sleep Phase for Memory Consolidation

Mimics the brain's memory consolidation during sleep (Synaptic Homeostasis Hypothesis).

- Organize with local rules without seeing the whole picture
- Global downscaling (uniform decay)
- Recently accessed items offset the decay

### 3. Structuring by Perspective

Each agent has perspectives according to its role, storing learnings in a structured manner.

```json
{
  "agent_id": "procurement_agent_01",
  "perspectives": ["Cost", "Delivery", "Supplier", "Quality", "Alternatives"],
  "strength_by_perspective": {
    "Cost": 2.1,
    "Delivery": 0.8,
    "Supplier": 1.5
  }
}
```

### 4. Two-Stage Search Structure

```
Stage 1: Relevance Filter (Vector Search)
    → Exclude irrelevant information

Stage 2: Priority Ranking (Score Synthesis)
    → Similarity 0.40 + Strength 0.40 + Recency 0.20
```

### 5. Education Process

Form expertise through the same process as human education:

```
Textbook → Read in chunks → Test → Sleep (decay) → Repeat
```

### 6. Specialized Agent Constraint: 1 Task = 1 Sleep

Specialized agents do NOT have intermediate sleep. 1 task = 1 session = 1 sleep.

```
Orchestrator:
├── Manages multiple subtasks
├── Has intermediate sleep (context 70%, idle 1 hour, etc.)
└── Can save progress state to external storage and resume

Specialized Agent:
├── 1 task = 1 session = 1 sleep
├── No intermediate sleep
├── Compaction = Task failure
└── Orchestrator responsible for splitting tasks to appropriate size
```

### 7. Progress Monitoring and Early Abort

Orchestrator monitors agent progress and intervenes before compaction.

```
Progress Monitoring Flow:
Delegate task to agent
    ↓
Periodic progress checks (e.g., every 10 minutes)
    ↓
Judgment:
├── On track → Continue
├── Danger (progress < context usage) → Instruct early abort
└── Blocked → Report to user
```

### 8. Progress State External Recording

Orchestrator records progress to external file on task instruction/result receipt.

- Users can check progress at any time
- Recovery possible after orchestrator intermediate sleep
- Manages task dependency tree (task_tree)
- Records artifact file paths for handoff to subsequent tasks

## Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Input Processing Layer (Lightweight LLM: Haiku, etc.)      │
├─────────────────────────────────────────────────────────────┤
│  Large input → Mechanical split → Extract points → Summary  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator                                                │
├─────────────────────────────────────────────────────────────┤
│  Read only summary for routing decisions                    │
│  Dedicated memory: Agent expertise, assignment history, load│
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
│  Sleep Phase (Per Task Completion)                           │
├─────────────────────────────────────────────────────────────┤
│  Task complete → Extract learnings → Decay → Archive → Sleep │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Progress State File (External Record)                       │
├─────────────────────────────────────────────────────────────┤
│  Task tree, dependencies, artifact paths, issue/decision log │
│  → User can check progress, used for recovery from sleep     │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

All implementable with existing technology. No innovative new tech required.

| Component | Existing Technology |
|-----------|---------------------|
| Vector DB | Pinecone, Qdrant, Chroma, etc. |
| Metadata Management | PostgreSQL, MySQL, etc. |
| LLM API | Claude, GPT, etc. |
| Periodic Batch Processing | cron, Cloud Scheduler, etc. |
| Orchestration | Python, LangGraph, etc. |

## Implementation Effort Estimate

| Phase | Content |
|-------|---------|
| PoC | Single agent, basic reinforcement/decay, verification |
| Production Level | Multiple agents, parameter tuning, error handling, monitoring |
| Education Process | Textbook structure design, test creation, review scheduling |

## Relationship with Google Titan

Google's Titan architecture has a mechanism to update parameters at test time, but is still in research stage.

This architecture is positioned as a "practical solution until Titan becomes production-ready". True "internalization of expertise" requires parameter updates, but external memory accumulation and strength management can achieve a **significantly better state than now**.

## Documentation

For detailed design, see [docs/architecture.md](docs/architecture.md).

## Background

This architecture design emerged through discussions with Claude Opus 4.5 on January 11, 2025. Through deep consideration of the fundamental limitations of LLM agents, and referencing insights from human organizations and neuroscience, we designed a solution implementable with current technology.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

This project is still at the idea stage. If you're interested in implementation or have feedback, please create an Issue.
