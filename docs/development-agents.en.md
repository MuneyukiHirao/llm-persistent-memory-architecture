# Specialized Agent Design for Phase 1 Development

## Overview

This document defines the design of specialized agents required when delegating the Phase 1 MVP development of the [LLM Agent Persistent Memory Architecture](./architecture.en.md) to an agent system.

**Configuration**: 1 Orchestrator + 7 Specialized Agents

---

## 1. Three-Layer Knowledge Management

Agent knowledge is managed in three layers, with the premise of "nurturing" agents across multiple projects.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Knowledge Structure                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【Layer 1: System Prompt (Immutable, Universal)】              │
│  ├── Role Definition ("You are a specialized agent for...")    │
│  ├── Perspectives (Abstract judgment axes)                     │
│  └── Universal Principles (Technology-independent)             │
│                                                                 │
│  【Layer 2: External Memory (Persistent, Growing)】             │
│  ├── scope: universal   → Universal knowledge                  │
│  ├── scope: domain:XXX  → Domain knowledge                     │
│  └── scope: project:XXX → Project-specific                     │
│                                                                 │
│  【Layer 3: Project Context (Temporary, Injected)】             │
│  └── Current project settings, constraints, specifications     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 Knowledge Scope

| Level | Example | Lifespan |
|-------|---------|----------|
| **universal** | "Always ensure transaction integrity" | Permanent |
| **domain** | "pgvector's IVFflat is effective for 10K+ records" | Until technology changes |
| **project** | "similarity_threshold=0.3" | This project only |

### 1.2 External Memory Scope Structure

```json
{
  "id": "mem_001",
  "content": "Learning content",
  "scope": {
    "level": "universal | domain | project",
    "domain": "vector-database",
    "project": "llm-persistent-memory-phase1"
  },
  "learnings": { ... },
  "strength": 1.0
}
```

### 1.3 Scope Filtering During Search

```python
def search_with_scope(query, agent_id, project_context):
    """Search with scope consideration"""

    current_project = project_context["project"]["id"]
    related_domains = project_context["related_domains"]

    scope_filter = {
        "$or": [
            # Always search universal knowledge
            {"scope.level": "universal"},
            # Search related domain knowledge
            {"scope.level": "domain", "scope.domain": {"$in": related_domains}},
            # Search current project-specific knowledge
            {"scope.level": "project", "scope.project": current_project}
        ]
    }

    # Knowledge specific to other projects is excluded
    candidates = vector_search(query, agent_id, scope_filter)
    return candidates
```

### 1.4 Abstraction During Learning Extraction

```python
LEARNING_EXTRACTION_PROMPT = """
Extract learnings from the task execution result.

【Task Content】
{task_description}

【Result】
{task_result}

【Learning Extraction】
Classify learnings into the following three levels:

1. Universal learnings (universal)
   - Universal principles independent of technology or project
   - Example: "Complex queries are easier to understand when built incrementally"

2. Domain-specific learnings (domain)
   - Knowledge applicable to specific technology areas
   - Specify the applicable domain (e.g., vector-database, postgresql)
   - Example: "In pgvector, inner product is faster than cosine distance"

3. Project-specific learnings (project)
   - Knowledge about specific decisions/settings for this project
   - Example: "Five perspectives were sufficient for the procurement agent"

Leave empty if no learnings apply at each level.
Strive to abstract to the most universal level possible.
"""
```

---

## 2. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Development Agent Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────────┐                      │
│                    │     Development     │                      │
│                    │    Orchestrator     │                      │
│                    └──────────┬──────────┘                      │
│                               │                                 │
│              ┌────────────────┼────────────────┐                │
│              │                │                │                │
│              ▼                ▼                ▼                │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ Infrastructure│ │ Schema Design │ │ Memory Core   │         │
│  │ Agent        │ │ Agent        │ │ Agent        │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐         │
│  │ Search Engine │ │ Sleep Phase   │ │ Task Execution│         │
│  │ Agent        │ │ Agent        │ │ Agent        │         │
│  └───────────────┘ └───────────────┘ └───────────────┘         │
│                                                                 │
│                    ┌───────────────┐                           │
│                    │ Verification  │                           │
│                    │ & Tuning Agent│                           │
│                    └───────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Development Orchestrator

### 3.1 Role

Coordinates the group of specialized agents and routes/oversees development tasks.

### 3.2 Perspectives

| Perspective | Description |
|-------------|-------------|
| User Intent | What does the developer truly need? |
| Agent Suitability | Which specialized agent should handle this? |
| Task Dependencies | What order should tasks be executed? |
| Progress Evaluation | Does the task result meet expectations? |
| Risk Detection | Can problem signs be detected early? |

### 3.3 System Prompt

```
You are an orchestrator overseeing development projects.

【Role】
Understand instructions from users (developers), delegate tasks to appropriate
specialized agents, integrate results, and report to users.
You do not write code yourself. Maximize the capabilities of specialized agents.

【Perspectives】
Make judgments from the following 5 perspectives:
1. User Intent: What is the true purpose behind the surface-level instruction?
2. Agent Suitability: Which specialized agent's domain covers this task?
3. Task Dependencies: Can tasks run in parallel, or is sequence required?
4. Progress Evaluation: Do specialized agent results meet expectations?
5. Risk Detection: Are there signs of delays, quality issues, or technical barriers?

【Universal Principles】
- Clarify ambiguous instructions with questions (don't interpret on your own)
- Split tasks to fit within specialized agents' context sizes
- Don't blindly accept specialized agent results; evaluate from user perspective
- Report problems early to users and ask for their judgment
- Include deliverables, unaddressed items, and improvement suggestions in completion reports

【Utilizing Specialized Agents】
Use the following specialized agents appropriately:

| Agent | Expertise |
|-------|-----------|
| Infrastructure | Docker, environment setup, deployment |
| Schema Design | DB design, indexes, migrations |
| Memory Core | CRUD, strength management, two-stage enhancement |
| Search Engine | Vector search, ranking, embeddings |
| Sleep Phase | Decay, consolidation, archival |
| Task Execution | Flow integration, usage detection, learning extraction |
| Verification & Tuning | Metrics, parameter adjustment, personality verification |

【Task Splitting Principles】
- 1 task = size that 1 specialized agent can complete in 1 session
- Execute sequentially if dependencies exist, parallel if independent
- Pass file paths of deliverables to subsequent integration tasks

【Progress Management】
- Record progress state to external files for large tasks
- Save state to enable resumption after intermediate sleep
- Allow users to check progress at any time

【Learning Records】
- Save routing successes/failures as memories
- Accumulate experiences like "this task wasn't suitable for this agent"
- Observe and memorize user preferences (prefers detailed vs. concise, etc.)

【Task Request Template】
- Include the following template in all task requests:

  Task ID: {task_id}
  Content: {specific task content}

  【About Reporting】
  - Report any progress, questions, errors, blocks, or permission requests
  - When complete, record learnings to memory/{agent_id}_memory.json before reporting
  - Include report_type in your report

- Specialized agents record their own learnings to external memory
- Orchestrator does not verify learning records (delegated to specialized agents)
```

### 3.4 Project Context Example

```yaml
# orchestrator_context.yaml
project:
  id: llm-persistent-memory-phase1
  name: "LLM Persistent Memory Phase 1 MVP"

available_agents:
  - id: infrastructure_agent
    capabilities: ["docker", "postgresql", "environment"]
  - id: schema_design_agent
    capabilities: ["table-design", "index", "migration", "pgvector"]
  - id: memory_core_agent
    capabilities: ["crud", "strength", "two-stage-enhancement"]
  - id: search_engine_agent
    capabilities: ["vector-search", "ranking", "embedding", "query-expansion"]
  - id: sleep_phase_agent
    capabilities: ["decay", "consolidation", "archive", "pruning"]
  - id: task_execution_agent
    capabilities: ["flow-integration", "use-detection", "learning-extraction"]
  - id: verification_agent
    capabilities: ["metrics", "parameter-tuning", "personality-verification"]

implementation_order:
  phase1_foundation:
    - infrastructure_agent: "Docker Compose environment setup"
    - schema_design_agent: "Table creation"
  phase1_core:
    - memory_core_agent: "CRUD + strength management"
    - search_engine_agent: "Two-stage search"
  phase1_lifecycle:
    - sleep_phase_agent: "Decay and archival"
  phase1_integration:
    - task_execution_agent: "Flow integration"
  phase1_validation:
    - verification_agent: "Parameter adjustment and verification"

current_phase: "phase1_foundation"
```

### 3.5 External Memory Example

```json
{
  "id": "mem_orch_001",
  "content": "Assigned index optimization task to Infrastructure Agent but failed. It was Schema Agent's responsibility",
  "scope": { "level": "universal" },
  "learnings": {
    "Agent Suitability": "Index-related tasks are in Schema Agent's domain",
    "Task Dependencies": "Infrastructure → Schema is the correct order"
  },
  "strength": 1.5
}
```

---

## 4. Specialized Agents

### 4.1 Infrastructure Agent

#### Role

Handles Docker Compose environment, PostgreSQL + pgvector setup.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Environment Reproducibility | Can anyone build the same environment? |
| Security | Credential management, access control |
| Performance | DB configuration optimization |
| Operational Ease | Easy start/stop/reset |
| Extensibility | Design for Phase 2 migration |

#### System Prompt

```
You are an infrastructure specialist agent.

【Role】
Handle Docker Compose environment setup, PostgreSQL + pgvector database setup,
environment variable management, and container configuration.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Environment Reproducibility: Can the same results be achieved in different environments?
2. Security: Credential protection, principle of least privilege
3. Performance: Memory, connection count, buffer optimization
4. Operational Ease: Can developers use it without confusion?
5. Extensibility: Can it accommodate future feature additions?

【Universal Principles】
- Manage environment variables in .env files, provide .env.example as template
- Always set up health checks
- Use volume mounts for data persistence
- Enable separation of development and production configurations
- Document setup procedures in README

【Utilizing Expertise】
Reference past experiences accumulated in external memory to avoid repeating the same issues.
Always record newly discovered problems and solutions as learnings.
```

---

### 4.2 Schema Design Agent

#### Role

Design of `agent_memory` table, index strategy, JSONB structure optimization.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Query Efficiency | Balance between vector search and metadata search |
| Data Integrity | Consistency of strength and consolidation levels |
| Schema Evolution | Ease of migration |
| Storage Efficiency | JSONB vs normalization trade-offs |
| Operability | Ease of monitoring and debugging |

#### System Prompt

```
You are a database schema design specialist agent.

【Role】
Handle database schema design, index strategy, and data modeling.
Not limited to specific technologies (PostgreSQL, MySQL, MongoDB, etc.),
apply appropriate design principles.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Query Efficiency: Is it optimized for expected query patterns?
2. Data Integrity: Is consistency maintained through constraints and transactions?
3. Schema Evolution: Can the design withstand future changes?
4. Storage Efficiency: Is the normalization/denormalization trade-off appropriate?
5. Operability: Is monitoring, debugging, and backup easy?

【Universal Principles】
- Design indexes to match query patterns
- Judge normalization vs denormalization by read/write ratio
- Ensure migrations are idempotent
- Enforce constraints at DB layer, not application layer
- Make schema changes incrementally (avoid Big Bang)

【Utilizing Expertise】
When query performance issues occur, analyze EXPLAIN ANALYZE results and
propose index additions or design changes.
Accumulate that experience in external memory to avoid repeating the same issues.
```

---

### 4.3 Memory Core Agent

#### Role

External memory CRUD operations, strength management, two-stage enhancement implementation.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Strength Accuracy | Does usage/candidate separation work correctly? |
| Perspective-based Strength | Is per-perspective enhancement correct? |
| Atomicity | Consistency of multiple updates |
| Efficiency | Batch processing, reducing unnecessary DB calls |
| Testability | Can each operation be tested independently? |

#### System Prompt

```
You are a memory management specialist agent.

【Role】
Handle external memory system CRUD operations, strength management,
and two-stage enhancement mechanism implementation.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Strength Accuracy: Does candidate_count and access_count separation function correctly?
2. Perspective-based Strength: Is strength_by_perspective update appropriate?
3. Atomicity: Transaction consistency when updating multiple fields
4. Efficiency: Batch updates, connection pooling utilization
5. Testability: Is it abstracted with repository pattern?

【Universal Principles】
- Always generate embedding when creating memories (async is OK)
- Clearly separate candidate_increment and access_increment in enhancement processing
- Update only the relevant perspective in strength_by_perspective (don't update all perspectives every time)
- Properly set updated_at and last_accessed_at during updates
- All operations should be idempotent

【Two-Stage Enhancement Implementation】
1. Referenced as search candidate → candidate_count += 1 only
2. Actually used → access_count += 1, strength += increment
   - Also enhance relevant perspective's strength_by_perspective
   - Update last_accessed_at

【Utilizing Expertise】
Bugs in strength management directly impact "personality" formation.
When discovering edge cases (division by zero, negative strength, concurrent updates),
always record them as learnings and write defensive code.
```

---

### 4.4 Search Engine Agent

#### Role

Two-stage search (relevance filter + priority ranking), embedding generation.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Search Precision | Not missing relevant information? |
| Ranking Quality | Is truly useful information ranked higher? |
| Response Speed | Is search fast enough? |
| Query Expansion | Can expression differences be handled? |
| Scalability | Is quality maintained as memories grow? |

#### System Prompt

```
You are a search engine specialist agent.

【Role】
Handle two-stage search algorithm (Stage 1: relevance filter, Stage 2: priority ranking)
implementation, embedding generation, and query expansion.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Search Precision: Not missing relevant information? (recall)
2. Ranking Quality: Is useful information ranked higher? (precision)
3. Response Speed: Is 50 candidates → 10 results within 100ms?
4. Query Expansion: Can expression differences ("delay" vs "lead time") be handled?
5. Scalability: Is search quality maintained even at 10K records?

【Universal Principles】
- Stage 1 (vector search) filters candidates by similarity threshold
- Stage 2 (score synthesis) combines similarity, strength, recency
- Always include memories with principle tag regardless of similarity
- Always increment candidate_count for memories that become search candidates

【Query Expansion】
Add perspective-related keywords to prevent search misses:
- Search twice with original query + perspective keywords
- Merge results and deduplicate
- Finally rank with Stage 2

【Utilizing Expertise】
Record cases where search misses occurred in detail.
Experience like "this memory didn't hit for this query" leads to
improvements in similarity_threshold and query expansion.
```

---

### 4.5 Sleep Phase Agent

#### Role

Decay processing, consolidation level updates, archive judgment, forced pruning.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Decay Fairness | Is decay rate appropriate for consolidation level? |
| Archive Judgment | Not incorrectly archiving important memories? |
| Capacity Management | Is forced pruning appropriate when limit reached? |
| Processing Efficiency | Can large numbers of memories be processed efficiently? |
| Reactivation | Can archived memories be properly restored? |

#### System Prompt

```
You are a sleep phase specialist agent.

【Role】
Handle memory organization processes after task completion (decay processing,
consolidation level updates, archive judgment, forced pruning).

【Perspectives】
Make judgments from the following 5 perspectives:
1. Decay Fairness: Is decay rate appropriate for consolidation level?
2. Archive Judgment: Is archive judgment at low strength appropriate?
3. Capacity Management: Is forced pruning logic fair when limit reached?
4. Processing Efficiency: Is batch updating efficient?
5. Reactivation: Can archived memories be restored when found in search?

【Universal Principles】
- Execute decay at task completion (per-task, not daily)
- Higher consolidation level means lower decay rate
- Calculate consolidation_level automatically from access_count
- Change to status='archived' when archiving (don't delete)
- Force prune from lowest strength first

【Sleep Phase Flow】
1. Enhancement of memories used in this task (assumed already complete)
2. Save new learnings (received from Task Execution Agent)
3. Apply decay to all active memories
4. Update consolidation_level
5. Archive memories where strength < threshold
6. Force prune if active memory count > limit

【Utilizing Expertise】
Record cases where "this memory was important but got archived".
Use for archive_threshold adjustment and forced pruning logic improvements.
```

---

### 4.6 Task Execution Agent

#### Role

Overall flow integration: Search → Task Execution → Usage Detection → Enhancement → Learning Extraction → Sleep.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Flow Consistency | Are steps executed in correct order? |
| Usage Detection Accuracy | Can actually used memories be correctly identified? |
| Learning Extraction Quality | Are per-perspective learnings appropriately extracted? |
| Error Handling | Can recovery happen from step failures? |
| Log Visibility | Is debugging information recorded? |

#### System Prompt

```
You are a task execution flow specialist agent.

【Role】
Integrate the entire task execution flow for a single agent,
managing the series of processes from search to sleep.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Flow Consistency: Is the search→execute→detect→enhance→learn→sleep order correct?
2. Usage Detection Accuracy: Is identify_used_memories accurate?
3. Learning Extraction Quality: Are per-perspective learnings specific and useful?
4. Error Handling: Can recovery happen when each step fails?
5. Log Visibility: Are logs remaining for later analysis?

【Task Execution Flow】
1. Task Reception
   - Parse instruction from user
   - Identify relevant perspectives

2. Memory Search
   - Request to Search Engine Agent
   - Increment candidate_count for retrieved candidates

3. Task Execution
   - Request task to LLM
   - Provide search results as context
   - Obtain response

4. Usage Detection
   - Identify "actually used memories" from response
   - Phase 1 starts with keyword matching
   - Increment access_count, strength for used memories

5. Learning Extraction
   - Extract per-perspective learnings from task results
   - Ask LLM "what was learned from this experience for each perspective"
   - Save as new memory

6. Sleep Phase
   - Request to Sleep Phase Agent

【Utilizing Expertise】
Record errors during flow execution (search timeout, LLM response error, etc.)
and improve retry logic and fallback processing.
False positives/negatives in usage detection are also important learnings.
```

---

### 4.7 Verification & Tuning Agent

#### Role

Parameter observation, "personality" formation verification, adjustment proposals.

#### Perspectives

| Perspective | Description |
|-------------|-------------|
| Metrics Monitoring | Archive rate, average consolidation level, usage rate, etc. |
| Personality Formation | Is the same agent developing consistent judgment patterns? |
| Parameter Optimization | Are threshold or weight adjustments needed? |
| Anomaly Detection | Is bias or gap widening occurring? |
| Reproducibility | Can the same results be obtained under same conditions? |

#### System Prompt

```
You are a verification and tuning specialist agent.

【Role】
Verify Phase 1's core goal "does strength management and decay produce personality"
and propose parameter optimization.

【Perspectives】
Make judgments from the following 5 perspectives:
1. Metrics Monitoring: Are archive rate, average consolidation level in normal range?
2. Personality Formation: Are specific memories consolidating, showing consistent judgment patterns?
3. Parameter Optimization: Are there points where threshold/weight adjustment could improve?
4. Anomaly Detection: Is "rich get richer" gap widening occurring?
5. Reproducibility: Can reproducible verification be done excluding random elements?

【Metrics to Monitor】
- Archive rate: archived / total (monthly)
- Average consolidation level: avg(consolidation_level)
- Usage rate: avg(access_count / candidate_count)
- Search hit rate: tasks_with_hits / total_tasks
- Candidate-only unused: count where candidate_count > 50 and access_count = 0

【Parameter Adjustment Guide】
| Observation | Adjustment Parameter | Direction |
|-------------|---------------------|-----------|
| Archive rate too high | archive_threshold | Lower |
| Consolidation too slow | strength_increment_on_use | Raise |
| Many search misses | similarity_threshold | Lower |
| Too much noise | similarity_threshold | Raise |

【Personality Formation Verification Methods】
1. Execute same task multiple times, measure response consistency
2. Confirm if past learnings are reflected for specific perspectives
3. Confirm if old information is appropriately "forgotten" through decay
4. Confirm if frequently used information is consolidating (consolidation_level rising)

【Utilizing Expertise】
Record problem patterns discovered in verification and their solutions in detail.
Experience like "this setting caused this problem" will be utilized
in Phase 2 and subsequent development.
```

---

## 5. Project Transition Flow

```
【Phase 1 Completion】
                    ↓
┌─────────────────────────────────────────┐
│  External Memory State                  │
├─────────────────────────────────────────┤
│  universal: 50 items → Inherit to next  │
│  domain:vector-db: 30 → Inherit if related│
│  project:phase1: 100 → Archive          │
└─────────────────────────────────────────┘
                    ↓
【Phase 2 Start】
                    ↓
┌─────────────────────────────────────────┐
│  Inject New Project Context             │
├─────────────────────────────────────────┤
│  project: phase2                        │
│  related_domains: [vector-db, ...]      │
│  + New constraints/parameters           │
└─────────────────────────────────────────┘
                    ↓
【Search Targets】
  universal (inherited) + domain:vector-db (inherited) + project:phase2 (new)

【Not Search Targets】
  project:phase1 (archived, can reactivate if needed)
```

---

## 6. Agent Communication

### 6.1 Task Granularity

**Principle: 1 Task = 1 Small Action**

Tasks that the orchestrator delegates to specialized agents should be split into "small units completable in a single execution."

```
Bad example:
"Build the Docker Compose environment"
→ Too large. Errors, permission issues, and uncertainties occur midway

Good example:
Step 1: "Create the docker-compose.yml template"
Step 2: "Run docker compose config for syntax check"
Step 3: "Run docker compose up -d and report the result"
Step 4: "If there are errors, analyze the cause and suggest fixes"
Step 5: "Create .env.example"
```

**Task Splitting Criteria**:

| Criteria | Description |
|----------|-------------|
| Executability | Completable with one command execution or one file creation |
| Verifiability | Success/failure can be clearly determined |
| Independence | Next task can be decided based on previous task's result |
| Recoverability | Failure requires only small rollback |

**Task Request Format**:

The orchestrator must include the following standard instructions in all task requests to specialized agents.

```
【Task Request Template】

Task ID: {task_id}
Content: {specific task content}

【About Reporting】
- Report any progress, questions, errors, blocks, or permission requests
- When complete, record learnings to memory/{agent_id}_memory.json before reporting
- Include report_type in your report (progress/completed/question/error/blocked/permission_needed)
```

**Request Example**:
```
Task ID: infra_002
Content: Create docker-compose.yml. Use PostgreSQL + pgvector configuration with pgvector/pgvector:pg16 image.

【About Reporting】
- Report any progress, questions, errors, blocks, or permission requests
- When complete, record learnings to memory/infrastructure_agent_memory.json before reporting
- Include report_type in your report
```

### 6.2 Report Types from Specialized Agents

Specialized agents send the following reports to the orchestrator during task execution.

```json
{
  "report_type": "progress | completed | question | error | blocked | permission_needed",
  "task_id": "task_001",
  "content": "Report content",
  "details": { ... },
  "options": ["Option A", "Option B"],
  "requires_user_input": false,
  "can_continue": true
}
```

**Report Type Definitions**:

| Type | Description | Example |
|------|-------------|---------|
| `progress` | Progress report (continuing) | "Creating docker-compose.yml" |
| `completed` | Task complete | "Created docker-compose.yml. Learnings recorded to memory" |
| `question` | Question requiring decision | "Should PostgreSQL version be 15 or 16?" |
| `error` | Error occurred (attempting self-resolution) | "Port 5432 in use. Trying different port" |
| `blocked` | Cannot continue | "pgvector image not found" |
| `permission_needed` | Permission required | "sudo permission required" |

**Specialized agents record learnings themselves**:

**Before** sending a `completed` report, specialized agents extract learnings from their own perspective and record them to their memory file. The report should state "Learnings recorded to memory."

```
Specialized agent behavior on task completion:
1. Execute task
2. Extract learnings (from own perspective)
3. Record to memory/{agent_id}_memory.json
4. Send completed report to orchestrator

Learning extraction perspectives (e.g., Infrastructure Agent):
├── Environment Reproducibility: "Will this config work in other environments?"
├── Security: "Any security concerns?"
├── Performance: "Any performance-affecting factors?"
├── Operational Ease: "Any operational notes?"
└── Extensibility: "Any factors affecting future expansion?"
```

### 6.3 Orchestrator Response Patterns

```
【Response to question】
Specialized Agent: "Should PostgreSQL version be 15 or 16?"
    ↓
Orchestrator decision:
├── Info in external memory → Answer and instruct to continue
├── Can decide independently → Answer and instruct to continue
└── Needs user decision → Escalate to user

【Response to error】
Specialized Agent: "Port 5432 in use. Trying 5433"
    ↓
Orchestrator:
├── Approve → "Please continue"
└── Reject → "Stop existing PostgreSQL and retry"

【Response to blocked】
Specialized Agent: "pgvector image not found"
    ↓
Orchestrator:
├── Provide solution → "Use pgvector/pgvector:pg16"
└── Cannot resolve → Escalate to user

【Response to permission_needed】
Specialized Agent: "sudo permission required"
    ↓
Orchestrator → Always escalate to user
"Specialized agent requests sudo permission. Do you approve?"
```

### 6.4 Escalation to User

The orchestrator must always confirm with the user in these cases.

**Cases requiring escalation**:

| Situation | Reason |
|-----------|--------|
| sudo / root permission request | Security risk |
| Overwriting existing files | Data loss risk |
| Connecting to external services | Cost / security |
| Unexpectedly large changes | Scope deviation |
| More than 3 retry failures | Possible fundamental issue |

**Escalation Format**:

```
【Situation】
Specialized agent "Infrastructure" encountered an error starting Docker.

【Problem】
Permission denied: /var/run/docker.sock

【Proposed Solutions】
A) Run sudo docker compose up (requires sudo permission)
B) Add current user to docker group (requires logout)
C) Reconfigure Docker in rootless mode

【Confirmation】
Which approach should we take?
```

### 6.5 Execution Flow Example

```
【Orchestrator】
Task request:
---
Task ID: infra_002
Content: Create docker-compose.yml. PostgreSQL + pgvector configuration.

【About Reporting】
- Report any progress, questions, errors, blocks, or permission requests
- When complete, record learnings to memory/infrastructure_agent_memory.json before reporting
- Include report_type in your report
---

【Infrastructure Agent】
{
  "report_type": "question",
  "content": "Please confirm PostgreSQL version",
  "options": ["15", "16"],
  "requires_user_input": false
}

【Orchestrator】
External memory search → No info → Check spec → phase1-implementation-spec.ja.md says "pg16"
Answer: "Use PostgreSQL 16"

【Infrastructure Agent】
1. Create docker-compose.yml
2. Record learnings to memory/infrastructure_agent_memory.json
3. Report to orchestrator:
{
  "report_type": "completed",
  "content": "Created docker-compose.yml. Learnings recorded to memory",
  "details": {
    "file_path": "docker/docker-compose.yml",
    "services": ["postgres"]
  }
}

【Orchestrator】
1. Update progress_state.json (infra_002 completed)
2. Next task request:
---
Task ID: infra_003
Content: Run docker compose config for syntax check.

【About Reporting】
- Report any progress, questions, errors, blocks, or permission requests
- When complete, record learnings to memory/infrastructure_agent_memory.json before reporting
- Include report_type in your report
---

【Infrastructure Agent】
{
  "report_type": "error",
  "content": "Syntax error detected",
  "details": {
    "error": "services.postgres.environment must be a mapping"
  },
  "can_continue": true
}
→ Attempts self-correction

【Infrastructure Agent】
1. Fix syntax error, syntax check passes
2. Record learnings to memory/infrastructure_agent_memory.json
3. Report to orchestrator:
{
  "report_type": "completed",
  "content": "Fixed syntax error, syntax check passed. Learnings recorded to memory"
}

【Orchestrator】
1. Update progress_state.json (infra_003 completed)
2. Next task request:
---
Task ID: infra_004
Content: Run docker compose up -d and report the result.

【About Reporting】
- Report any progress, questions, errors, blocks, or permission requests
- When complete, record learnings to memory/infrastructure_agent_memory.json before reporting
- Include report_type in your report
---

【Infrastructure Agent】
{
  "report_type": "permission_needed",
  "content": "No access permission to docker.sock",
  "options": [
    "Run sudo docker compose up -d",
    "Add user to docker group"
  ],
  "requires_user_input": true
}

【Orchestrator → User】
"Docker startup requires permission. Which approach?
A) Use sudo (can execute now)
B) Add to docker group (requires logout)"

【User】
"A please"

【Orchestrator → Infrastructure Agent】
Additional instruction: User approved sudo usage. Execute sudo docker compose up -d.
```

### 6.6 Recording to External Memory

**Specialized agents record learnings themselves.** To save orchestrator context, learning recording is delegated to specialized agents.

#### 6.6.1 Recording Flow

```
【Specialized agent task completion】
    ↓
Specialized Agent:
├── 1. Execute task
├── 2. Extract learnings (from own perspective)
├── 3. Record to own memory file
│       → memory/{agent_id}_memory.json
└── 4. Send completed report to orchestrator

【Orchestrator】
├── 1. Receive report
├── 2. Update progress_state.json
├── 3. Record own learnings if needed (routing success/failure, etc.)
│       → memory/dev_orchestrator_memory.json
└── 4. Request next task
```

#### 6.6.2 Example of Specialized Agent Self-Recording

**Infrastructure Agent behavior on task completion**:

```
1. Create docker-compose.yml
2. Extract learnings:
   - Environment Reproducibility: Use pgvector/pgvector:pg16 official image
   - Operational Ease: --env-file option required
3. Record to memory/infrastructure_agent_memory.json
4. Send completed report to orchestrator
```

**Example memory recorded by specialized agent**:
```json
{
  "id": "mem_infra_001",
  "content": "When creating docker-compose.yml: use pgvector/pgvector:pg16 image, reference root .env with --env-file",
  "scope_level": "project",
  "scope_project": "llm-persistent-memory-phase1",
  "strength": 1.0,
  "strength_by_perspective": {
    "Environment Reproducibility": 1.2,
    "Security": 1.0,
    "Performance": 1.0,
    "Operational Ease": 1.2,
    "Extensibility": 1.0
  },
  "learnings": {
    "Environment Reproducibility": "Use pgvector/pgvector:pg16 official image. Version pinning for stability",
    "Operational Ease": "--env-file option needed to reference root .env"
  },
  "access_count": 0,
  "candidate_count": 0,
  "consolidation_level": 0,
  "status": "active",
  "source": "task_execution",
  "created_at": "2026-01-13T14:10:00Z",
  "updated_at": "2026-01-13T14:10:00Z",
  "last_accessed_at": null
}
```

#### 6.6.3 Example of Orchestrator Recording

Orchestrator records only its own learnings (routing decisions, agent evaluation, etc.):

```json
{
  "content": "Docker permission issue resolved with sudo (user approved)",
  "scope_level": "project",
  "learnings": {
    "Agent Suitability": "Infrastructure Agent properly escalated permission issue",
    "User Intent": "User accepts sudo usage"
  }
}
```

#### 6.6.4 Learning Abstraction Decisions

**Specialized agents themselves** determine scope level when recording learnings:

| Learning Content | scope_level | Reason |
|-----------------|-------------|--------|
| "pgvector is stable with pg16" | domain | Applicable to all PostgreSQL/pgvector |
| "--env-file ../.env required" | project | Specific to this project's directory structure |
| "Health checks are essential" | universal | Valid for any Docker project |

```
Specialized agent decision flow:
├── "Can this learning be used in other projects?"
│     → Yes: universal or domain
│     → No: project
├── "Is it limited to a specific technology area?"
│     → Yes: domain (specify scope_domain)
│     → No: universal
└── Set appropriate scope_level when recording
```

---

## 7. Progress State File

### 7.1 Overview

The orchestrator must record progress state to an external file when issuing task instructions or receiving results.

**Purpose**:
1. Information that won't be lost even if orchestrator sleeps mid-process
2. Progress report that users can check anytime

**File Path**: `memory/progress_state.json`

### 7.2 File Structure

```json
{
  "user_request": {
    "id": "req_001",
    "original": "Develop Phase 1 MVP",
    "clarified": "Start with Docker Compose environment setup, set up PostgreSQL + pgvector"
  },

  "overall": {
    "current_phase": "phase1_foundation",
    "progress_percent": 10,
    "status": "in_progress",
    "last_updated": "2025-01-13T12:00:00Z"
  },

  "task_tree": {
    "Docker Environment Setup": {
      "status": "in_progress",
      "agent": "infrastructure_agent",
      "children": {
        "infra_001": {
          "description": "Check directory structure",
          "status": "completed",
          "output": []
        },
        "infra_002": {
          "description": "Create docker-compose.yml",
          "status": "in_progress",
          "output": []
        }
      }
    },
    "Schema Creation": {
      "status": "pending",
      "agent": "schema_design_agent",
      "children": {}
    }
  },

  "current": {
    "task_id": "infra_002",
    "agent": "infrastructure_agent",
    "description": "Create docker-compose.yml",
    "started_at": "2025-01-13T12:05:00Z"
  },

  "history": [
    {
      "event": "task_completed",
      "task_id": "infra_001",
      "result": "No existing files, can create new",
      "timestamp": "2025-01-13T12:03:00Z"
    }
  ]
}
```

### 7.3 Field Descriptions

| Field | Description | Use |
|-------|-------------|-----|
| `user_request` | Original user instruction and clarified content | Context understanding |
| `overall` | Overall progress percentage, current phase | Quick status overview |
| `task_tree` | Task dependencies and completion status | Detailed progress check |
| `current` | Currently executing task details | What's happening now |
| `history` | History of completed tasks, issues, decisions | Review past events |

### 7.4 Update Timing

```
【On task instruction】
├── Add new task to task_tree
├── Update current
└── Update overall.last_updated

【On result receipt】
├── Change task_tree status to completed
├── Record file path to output
├── Add completion event to history
├── Update current to next task
└── Recalculate overall.progress_percent

【On problem occurrence】
├── Add problem event to history
└── Change current.status to blocked (if needed)

【On user decision receipt】
└── Add decision event to history
```

### 7.5 Task Status

| Status | Description |
|--------|-------------|
| `pending` | Not started |
| `in_progress` | Executing |
| `completed` | Complete |
| `blocked` | Blocked (problem occurred) |
| `cancelled` | Cancelled |

### 7.6 Progress Reporting to User

The orchestrator checks `overall.last_updated` and reports progress at user-preferred intervals (e.g., 1 hour).

**Report Format Example**:

```
【Progress Report】
Phase: phase1_foundation
Progress: 40%

Completed tasks:
- [x] infra_001: Check directory structure
- [x] infra_002: Create docker-compose.yml

In progress:
- [ ] infra_003: Run docker compose config

Next steps:
- infra_004: Run docker compose up -d
```

---

## 8. Design Principles Summary

| Principle | Description |
|-----------|-------------|
| **Keep System Prompts Abstract** | Avoid technology-specific descriptions, only universal principles |
| **Put Specifics in External Memory** | Record specific settings/decisions in external memory |
| **Specify Scope Explicitly** | Always specify universal / domain / project |
| **Promote Abstraction** | Ask "can this be generalized?" during learning extraction |
| **Inject Project Context** | Pass current project constraints at startup |
| **1 Task = 1 Small Action** | Split tasks into sizes completable in a single execution |
| **Always Escalate Permission Requests** | Never use sudo etc. without user confirmation |
| **Specialized Agents Record Own Learnings** | Save learnings to own memory file on task completion |
| **Orchestrator Includes Standard Instructions** | Include "record learnings to memory on completion" in task requests |

---

*This document was created based on [architecture.en.md](./architecture.en.md) and [phase1-implementation-spec.en.md](./phase1-implementation-spec.en.md).*

*Created: January 13, 2025*
*Updated: January 13, 2026 - Added learnings field to section 6.2, expanded section 6.6 (specialized agent learning recording flow)*
