# Phase 1 Implementation Specification: Persistent Memory Architecture MVP

## Overview

This document defines the implementation specification for Phase 1 MVP (Minimum Viable Product) based on the [LLM Agent Persistent Memory Architecture Design](./architecture.ja.md).

**Verification Goal**: Validate whether strength management and decay can generate "personality"

---

## 1. MVP Scope

### 1.1 Phase Division

| Phase | Verification Target | Implementation Scope |
|-------|--------------------|--------------------|
| **Phase 1** | Whether strength management and decay generate "personality" | Single agent + external memory |
| Phase 2 | Whether orchestration functions properly | Multiple agents + routing |

### 1.2 Core Functions to Verify in Phase 1

1. **Two-Stage Reinforcement**
   - Separation between becoming a search candidate (candidate_count++) and actual usage (access_count++, strength++)
   - Prevents noise reinforcement, retaining only truly useful information

2. **Perspective-Based Strength**
   - Same information has different importance depending on perspective
   - Perspective-specific strength management via `strength_by_perspective`

3. **Sleep Phase**
   - Decay processing at task completion
   - Differentiated decay rates based on consolidation level
   - Archive determination

### 1.3 Items NOT Implemented in Phase 1

- Orchestrator (Phase 2)
- Routing between multiple agents (Phase 2)
- Input processing layer (Phase 2)
- Learnable neural scorer (start with linear scoring)
- Weight-based capacity management (start with count limit)
- Agent definition in DB (start with code definition)

---

## 2. Technology Stack

| Component | Technology Selection | Notes |
|-----------|---------------------|-------|
| Vector DB + Metadata | Azure Database for PostgreSQL + pgvector | Single DB manages vectors and metadata |
| Embedding | text-embedding-3-small (1536 dimensions) | Balance of cost efficiency and performance |
| LLM | Claude Sonnet 4 | Used for task execution and learning extraction |
| Execution Environment | Local (Docker Compose) | For development and validation |
| Language | Python (raw SDK) | anthropic SDK, psycopg2, openai (for embedding) |

### 2.1 Docker Compose Configuration

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agent_memory
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://agent:${POSTGRES_PASSWORD}@postgres:5432/agent_memory
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - postgres
```

---

## 3. Data Schema

### 3.1 Main Table: agent_memory

```sql
-- PostgreSQL + pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE agent_memory (
    -- Identifiers
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(64) NOT NULL,

    -- Content
    content TEXT NOT NULL,
    embedding vector(1536),
    tags TEXT[] DEFAULT '{}',

    -- Strength Management
    strength FLOAT DEFAULT 1.0,
    strength_by_perspective JSONB,  -- {"cost": 1.2, "delivery": 0.8, ...}

    -- Usage Tracking
    access_count INT DEFAULT 0,
    candidate_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP,

    -- Impact
    impact_score FLOAT DEFAULT 0.0,

    -- Consolidation Management
    consolidation_level INT DEFAULT 0,  -- 0-5

    -- Learnings (by perspective)
    learnings JSONB,  -- {"cost": "...", "delivery": "...", ...}

    -- Status
    status VARCHAR(16) DEFAULT 'active',  -- active / archived
    source VARCHAR(32),  -- education / task / manual

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_decay_at TIMESTAMP  -- Sleep phase processing tracking
);

-- Indexes
CREATE INDEX idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX idx_agent_memory_status ON agent_memory(status);
CREATE INDEX idx_agent_memory_tags ON agent_memory USING GIN(tags);
CREATE INDEX idx_agent_memory_strength ON agent_memory(strength);

-- Phase 1: No vector index (assuming less than 10,000 records)
-- Add the following when exceeding 10,000 records:
-- CREATE INDEX idx_agent_memory_embedding ON agent_memory
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 3.2 Perspective-Based Data Storage (JSONB Denormalization)

```sql
-- strength_by_perspective example
'{
    "cost": 1.2,
    "delivery": 0.8,
    "supplier": 1.5,
    "quality": 0.3,
    "alternative": 1.2
}'::JSONB

-- learnings example
'{
    "cost": "Emergency procurement increased cost by 15%",
    "delivery": "2-week buffer is necessary",
    "supplier": "Supplier Y has single-site risk"
}'::JSONB
```

### 3.3 Agent Definition (Phase 1: Code Definition)

```python
# config/agents.py

AGENTS = {
    "procurement_agent_01": {
        "agent_id": "procurement_agent_01",
        "role": "Procurement Agent",
        "perspectives": ["cost", "delivery", "supplier", "quality", "alternative"],
        "system_prompt": """You are a procurement specialist agent.
You make decisions from the perspectives of supplier selection, cost analysis,
delivery management, quality evaluation, and alternative procurement.
Please utilize past experiences and learnings to support optimal procurement decisions.""",
    }
}

def get_initial_strength_by_perspective(agent_id: str) -> dict:
    """Generate initial strength_by_perspective based on agent's perspectives"""
    agent = AGENTS.get(agent_id)
    if not agent:
        return {}
    return {p: 1.0 for p in agent["perspectives"]}
```

### 3.4 Timestamp Design

| Column | Purpose |
|--------|---------|
| `created_at` | Memory creation timestamp |
| `updated_at` | Last update timestamp (strength changes, etc.) |
| `last_accessed_at` | Last usage timestamp (for recency calculation) |
| `last_decay_at` | Last decay processing timestamp (sleep phase tracking) |

---

## 4. Initial Parameter Values

### 4.1 Strength Management Parameters

```python
# === Initial Strength ===
INITIAL_STRENGTH = 1.0              # Initial strength for new memories
INITIAL_STRENGTH_EDUCATION = 0.5    # Memories just read during education process

# === Reinforcement Amount ===
STRENGTH_INCREMENT_ON_USE = 0.1           # Reinforcement on usage
PERSPECTIVE_STRENGTH_INCREMENT = 0.15     # Perspective-specific strength reinforcement

# === Thresholds ===
ARCHIVE_THRESHOLD = 0.1             # Archive when below this
REACTIVATION_STRENGTH = 0.5         # Initial strength on reactivation
```

**Design Rationale**:
- Initial strength 1.0 is a neutral starting point
- At 5%/day decay, reaches threshold 0.1 in approximately 20 days
- Reinforcement amount 0.1 compensates for 1 day of decay per use (0.95 × 1.1 ≒ 1.05)

### 4.2 Consolidation Level and Decay Rate

```python
# === Expected Tasks ===
EXPECTED_TASKS_PER_DAY = 10

# === Consolidation Level Thresholds (access_count) ===
CONSOLIDATION_THRESHOLDS = [0, 5, 15, 30, 60, 100]

# === Daily Decay Targets ===
DAILY_DECAY_TARGETS = {
    0: 0.95,    # Unconsolidated: 5%/day
    1: 0.97,    # Level 1: 3%/day
    2: 0.98,    # Level 2: 2%/day
    3: 0.99,    # Level 3: 1%/day
    4: 0.995,   # Level 4: 0.5%/day
    5: 0.998,   # Fully consolidated: 0.2%/day
}

# === Per-Task Decay Rate (auto-calculated) ===
def calculate_decay_rates(daily_targets: dict, tasks_per_day: int) -> dict:
    """Calculate per-task decay rate from daily decay targets"""
    return {
        level: target ** (1 / tasks_per_day)
        for level, target in daily_targets.items()
    }

# Results:
# Level 0: 0.9949 (≒ 0.995)
# Level 1: 0.9970
# Level 2: 0.9980
# Level 3: 0.9990
# Level 4: 0.9995
# Level 5: 0.9998
```

### 4.3 Search Parameters

```python
# === Stage 1: Relevance Filter ===
SIMILARITY_THRESHOLD = 0.3          # Minimum similarity threshold
CANDIDATE_LIMIT = 50                # Maximum candidates in Stage 1

# === Stage 2: Priority Ranking (Linear Score) ===
SCORE_WEIGHTS = {
    "similarity": 0.50,             # Similarity weight
    "strength": 0.30,               # Strength weight
    "recency": 0.20,                # Recency weight
}

# === Final Results ===
TOP_K_RESULTS = 10                  # Number of results to pass to context

# === Learnable Scorer Transition Threshold ===
MIN_TRAINING_SAMPLES = 100          # Consider neural net transition after this
```

**Design Rationale**:
- Similarity 0.3 is a loose threshold to exclude "clearly unrelated" items
- Similarity is prioritized most (0.50), strength is auxiliary (0.30)
- Linear scoring is sufficient for Phase 1

### 4.4 Impact Score

```python
# === Impact Addition Amount ===
IMPACT_USER_POSITIVE = 2.0          # Positive feedback from user
IMPACT_TASK_SUCCESS = 1.5           # Contributed to task success
IMPACT_PREVENTED_ERROR = 2.0        # Contributed to error prevention

# === Impact to Strength Conversion Rate ===
IMPACT_TO_STRENGTH_RATIO = 0.2      # Add impact × 0.2 to strength
```

### 4.5 Capacity Management

```python
# === Phase 1: Simple Count Limit ===
MAX_ACTIVE_MEMORIES = 5000          # Maximum active memory count

# === For Phase 2 and beyond ===
# MAX_TOTAL_WEIGHT = 10000
# CONSOLIDATION_WEIGHTS = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
```

### 4.6 Usage Detection Parameters

```python
# === Usage Detection (identify_used_memories) ===
USE_DETECTION_METHOD = "keyword"                # "keyword" | "similarity" | "llm"
USE_DETECTION_SIMILARITY_THRESHOLD = 0.3        # Threshold for similarity method
```

**Design Rationale**:
- Start with keyword matching in Phase 1 (low cost)
- Gradually transition to similarity → llm if accuracy is insufficient

### 4.7 Query Expansion

```python
# === Query Expansion ===
ENABLE_QUERY_EXPANSION = True

# === Perspective Keywords (defined per agent) ===
PERSPECTIVE_KEYWORDS = {
    "procurement_agent_01": {
        "cost": ["price", "expense", "budget", "TCO", "cost reduction", "unit price"],
        "delivery": ["lead time", "delay", "schedule", "deadline", "shipment"],
        "supplier": ["vendor", "source", "procurement partner", "supply chain"],
        "quality": ["defect", "inspection", "standard", "quality criteria", "yield"],
        "alternative": ["substitute", "second source", "redundancy", "backup"],
    }
}
```

### 4.8 Embedding Settings

```python
# === OpenAI Embedding ===
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
```

---

## 5. Integrated Configuration File

```python
# config/phase1_config.py

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Phase1Config:
    """Phase 1 MVP Parameter Configuration"""

    # === Strength Management ===
    initial_strength: float = 1.0
    initial_strength_education: float = 0.5
    strength_increment_on_use: float = 0.1
    perspective_strength_increment: float = 0.15
    archive_threshold: float = 0.1
    reactivation_strength: float = 0.5

    # === Decay ===
    expected_tasks_per_day: int = 10
    consolidation_thresholds: List[int] = field(
        default_factory=lambda: [0, 5, 15, 30, 60, 100]
    )
    daily_decay_targets: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.95,
            1: 0.97,
            2: 0.98,
            3: 0.99,
            4: 0.995,
            5: 0.998,
        }
    )

    # === Search ===
    similarity_threshold: float = 0.3
    candidate_limit: int = 50
    top_k_results: int = 10
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "similarity": 0.50,
            "strength": 0.30,
            "recency": 0.20,
        }
    )

    # === Impact ===
    impact_user_positive: float = 2.0
    impact_task_success: float = 1.5
    impact_prevented_error: float = 2.0
    impact_to_strength_ratio: float = 0.2

    # === Capacity ===
    max_active_memories: int = 5000

    # === Usage Detection ===
    use_detection_method: str = "keyword"  # "keyword" | "similarity" | "llm"
    use_detection_similarity_threshold: float = 0.3

    # === Query Expansion ===
    enable_query_expansion: bool = True

    # === Embedding ===
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    def get_decay_rate(self, consolidation_level: int) -> float:
        """Get per-task decay rate based on consolidation level"""
        daily_target = self.daily_decay_targets.get(consolidation_level, 0.95)
        return daily_target ** (1 / self.expected_tasks_per_day)

    def get_consolidation_level(self, access_count: int) -> int:
        """Calculate consolidation level from access_count"""
        level = 0
        for i, threshold in enumerate(self.consolidation_thresholds):
            if access_count >= threshold:
                level = i
        return level


# Default configuration instance
config = Phase1Config()
```

---

## 6. Observation Metrics and Parameter Tuning Guide

### 6.1 Metrics to Observe

| Metric | Calculation | Normal Range |
|--------|-------------|--------------|
| Archive Rate | archived / total | 10-30%/month |
| Average Consolidation Level | avg(consolidation_level) | 1.0-2.0 |
| Usage Rate | avg(access_count / candidate_count) | 0.1-0.3 |
| Search Hit Rate | tasks_with_hits / total_tasks | 0.7-0.9 |

### 6.2 Parameter Tuning Directions

| Observation | Parameter to Adjust | Direction |
|-------------|--------------------| ----------|
| Archive rate too high | `archive_threshold` | Lower |
| Archive rate too low | `archive_threshold` | Raise |
| Consolidation too slow | `strength_increment_on_use` | Raise |
| Consolidation too fast | `strength_increment_on_use` | Lower |
| Too many search misses | `similarity_threshold` | Lower |
| Too much noise | `similarity_threshold` | Raise |
| Context overflow | `top_k_results` | Reduce |
| Information shortage | `top_k_results` | Increase |

### 6.3 Adjusting EXPECTED_TASKS_PER_DAY

After starting production, observe actual task frequency for about one week and adjust `expected_tasks_per_day`.

```python
# Observation example
# 70 tasks in 1 week → 10 tasks/day → keep as is
# 140 tasks in 1 week → 20 tasks/day → change expected_tasks_per_day = 20
```

---

## 7. Next Steps

### 7.1 Phase 1 Implementation Order

1. **Infrastructure Setup**
   - Docker Compose environment setup
   - PostgreSQL + pgvector setup
   - Table creation

2. **Core Function Implementation**
   - Embedding generation
   - Vector search (Stage 1)
   - Score synthesis (Stage 2)
   - Strength update (two-stage reinforcement)

3. **Sleep Phase Implementation**
   - Consolidation level update
   - Decay processing
   - Archive processing

4. **Task Execution Flow Implementation**
   - Search → Task execution → Usage detection → Reinforcement → Learning extraction → Sleep

5. **Validation & Tuning**
   - Parameter adjustment
   - Verification of "personality" formation

### 7.2 Conditions for Transitioning to Phase 2

- Core functions of Phase 1 (two-stage reinforcement, perspective-based strength, sleep phase) are stable
- Appropriate parameter values are roughly determined
- Formation of "personality" is confirmed with a single agent

---

*This document is the Phase 1 MVP implementation specification created based on [architecture.ja.md](./architecture.ja.md).*

*Created: January 12, 2025*
