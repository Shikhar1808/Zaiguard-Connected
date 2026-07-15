# ZaiGuard — Alert Generation & Suppression Engine
## Architecture & Concept Reference Document

**Module Owner:** Alert Generation, Thresholding & Suppression Pipeline
**Position in System:** Between event classifiers (violence, fire, dog attack, trespassing, accident) and the dashboard
**Purpose of this document:** Single source of truth for the architecture of this module — what it does, why every component exists, and the underlying concepts behind each technology choice. Intended for personal reference, mentor discussions, interviews, and team integration.

---

## 1. Module Scope and Responsibility

This module is the **Alert Engine**. It is the decision-making layer that sits between raw model detections and the human operator. It does not do any computer vision, tracking, or classification — all of that is upstream and complete by the time an event reaches this module.

Given a raw detection event from any of the five event-specific classifiers (violence, fire, dog attack, trespassing, road accident), this module must:

1. Decide whether the event is confident enough to act on (**thresholding**, time-aware and zone-aware)
2. Assemble the event into a complete, self-describing record (**enrichment**)
3. Prevent the same ongoing incident from flooding the dashboard with repeated alerts (**burst deduplication**)
4. Check whether this type of event, from this camera, at this time, has been explicitly or semantically dismissed before (**suppression**)
5. Assign a severity level and route the surviving alert to the dashboard (**tiering**)
6. Learn from operator feedback so suppression improves over time (**feedback loop**)

Everything downstream of step 5 (dashboard rendering, notifications) and everything upstream of step 1 (detection, tracking, feature extraction) belongs to other modules. This module's job is to be the **gatekeeper and decision engine** in between.

---

## 2. The Finalized Architecture — End to End Flow

```
                    Raw Detection Event
                    (from any classifier: violence / fire /
                     dog_attack / trespassing / accident)
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │  LAYER 1 — Threshold Gate                │
        │  per-event-type, time-aware,             │
        │  zone-aware effective threshold          │
        └─────────────────────────────────────────┘
                            │ (confidence ≥ effective threshold)
                            ▼
        ┌─────────────────────────────────────────┐
        │  LAYER 2 — Event Enrichment              │
        │  assemble full AlertEvent object         │
        │  (pure data assembly, no decisions)      │
        └─────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │  LAYER 3 — Burst Deduplication (Redis)   │
        │  TTL-based key per (camera, zone, type)  │
        │  + hysteresis on confidence escalation   │
        └─────────────────────────────────────────┘
                            │ (not a duplicate of an active incident)
                            ▼
        ┌─────────────────────────────────────────┐
        │  LAYER 4 — Suppression Gate              │
        │  4A: Postgres exact rule lookup          │
        │  4B: Qdrant semantic ANN similarity      │
        │      search against dismissed alerts     │
        └─────────────────────────────────────────┘
                            │ (not suppressed)
                            ▼
        ┌─────────────────────────────────────────┐
        │  LAYER 5 — Tier Assignment + Routing     │
        │  CRITICAL / HIGH / MEDIUM / LOW           │
        │  emits DashboardAlert                     │
        └─────────────────────────────────────────┘
                            │
                            ▼
                       Dashboard
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
     Operator dismisses             Operator confirms
   ("not important")                  (genuine event)
              │                            │
              ▼                            ▼
   Write suppression rule          Log to alert_log
   to Postgres (with TTL)          (Postgres / TimescaleDB)
              +
   Embed alert description,
   write vector to Qdrant
   `dismissed_alerts` collection
```

This pipeline is **stateless in the processing path** — the Alert Engine process itself holds no persistent state between events. All state (thresholds, rules, embeddings, dedup keys) lives in Redis, Postgres, or Qdrant. If the process restarts, nothing is lost and no event is left half-processed. This is a deliberate fault-tolerance property.

---

## 3. Layer-by-Layer Breakdown

### Layer 1 — Threshold Gate

**What it does:** Compares the incoming detection confidence against a computed "effective threshold" that depends on the event type, the time of day, and the zone's risk profile.

```
effective_threshold = base_threshold(event_type)
                       × time_multiplier(hour_of_day)
                       × zone_risk_multiplier(zone_id)
```

**Why per-event-type thresholds:** Different event types have different cost profiles for false negatives vs false positives. A missed fire is catastrophic; a missed instance of someone lingering near a restricted door is low-stakes. So fire gets a low (sensitive) base threshold, trespassing gets a high (selective) one.

**Why time-aware:** The prior probability of an event changes drastically by hour. A person in a restricted zone at 3 AM is far more likely to be a genuine trespasser than the same detection at 3 PM. Rather than training separate day/night models, a simple multiplier adjusts sensitivity by time.

**Why zone-aware:** A "violence" detection near a gym during practice hours carries a different baseline likelihood of being a false positive than the same detection in a stairwell.

**Why not hardcoded:** Every value in this formula — base thresholds, time multipliers, zone multipliers — lives in a Postgres config table, editable from the dashboard without redeploying code. This is the difference between a system that needs an engineer to retune it and one operators can tune themselves.

| Event Type  | Base Threshold | Rationale |
|-------------|---------------|-----------|
| Fire        | 0.60          | Miss cost is catastrophic — stay sensitive |
| Violence    | 0.72          | Physical activity/sports cause false positives |
| Dog Attack  | 0.68          | Play vs attack is genuinely hard to distinguish |
| Trespassing | 0.78          | High base rate of legitimate movement |
| Accident    | 0.70          | Road context is relatively unambiguous |

---

### Layer 2 — Event Enrichment

**What it does:** Takes an event that passed the threshold gate and assembles it into a complete, standardized `AlertEvent` object containing every piece of context any downstream layer or teammate could need.

**Why it's a separate layer from thresholding:** Thresholding is a *decision*. Enrichment is *data assembly*. Separating them means each can be tested independently, and it establishes a single clean data contract — every layer after this one, and every teammate's module that consumes your output, only ever needs to understand one object shape: `AlertEvent`. This is the core of making integration painless.

`AlertEvent` fields:

```
alert_id          UUID   — generated here, deterministic (see Idempotency, §4.10)
pipeline          str    — "violence" | "fire" | "dog_attack" | "trespassing" | "accident"
raw_confidence    float  — straight from the model
effective_conf    float  — confidence after threshold computation context
camera_id         str
zone_id           str
zone_label        str    — human-readable, e.g. "gym_east"
timestamp         datetime (UTC)
hour_of_day       int
day_of_week       int
frame_ref         str    — path/reference to evidence clip (owned upstream)
involved_ids      List[int] — tracked object IDs
pipeline_features dict   — whatever event-specific features came from upstream
```

---

### Layer 3 — Burst Deduplication (Redis)

**What it does:** Prevents a single ongoing incident from generating one alert per video frame. A 30-second fight at 15fps would otherwise produce 450 near-identical alerts.

**Mechanism:**
- Compute `dedup_key = f"{camera_id}:{zone_id}:{event_type}"`
- On a new event: check Redis for this key.
  - If present → this is a duplicate of an active incident → drop, **unless** the new confidence exceeds the stored confidence by more than a configured jump (e.g. 0.15) — in which case let it through as an *escalation* (this is the hysteresis mechanism, see §4.8).
  - If absent → set the key with a TTL (event-type-specific, e.g. 45s for violence, 120s for fire) and let the event through.

**Why Redis specifically:** This check must run for *every single event*, potentially thousands per second across many cameras. Redis lives entirely in RAM — reads/writes take roughly 100 nanoseconds, versus ~100 microseconds for an SSD-backed database — a ~1000x difference. Doing this check in Postgres would mean a write+read on every event, which becomes a bottleneck under load. Redis's native TTL feature (automatic key expiry with zero cleanup code) is exactly the primitive this layer needs.

---

### Layer 4 — Suppression Gate

This is the core deliverable of the module — the mechanism that directly addresses "alert fatigue," the central problem statement of the entire project.

#### 4A — Exact Rule Store (Postgres)

**What it does:** Checks for explicit, structured suppression rules an operator has previously configured (manually, or auto-promoted from repeated dismissals).

**Schema:**

```sql
CREATE TABLE suppression_rules (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id   TEXT NOT NULL,
    zone_id     TEXT,
    event_type  TEXT NOT NULL,
    hour_start  INT,
    hour_end    INT,
    days_mask   INT,            -- bitmask, see §4.9
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ,     -- NULL = permanent
    source      TEXT             -- "manual" | "auto_promoted"
);

CREATE INDEX ON suppression_rules (camera_id, event_type);
```

**Why exact rules run first:** SQL index lookup (~1ms) is cheaper than vector ANN search (~5ms). Cheap, high-confidence checks should always run before expensive, probabilistic ones. Also, an explicit operator-configured rule is a deliberate decision and should take precedence over a similarity heuristic.

**Why TTL on rules:** A dismissal at 3 AM shouldn't silently suppress a genuine event at 3 PM the next day unless the operator explicitly intended permanence. Default dismissals get a 24-hour `expires_at`; an explicit "always ignore" checkbox sets `expires_at = NULL`.

#### 4B — Semantic Similarity Store (Qdrant)

**What it does:** Catches false-positive patterns that are *similar but not identical* to previously dismissed alerts — cases the exact rule store would miss (slightly different time, adjacent zone, etc.).

**Mechanism:**
1. Build a natural-language description of the `AlertEvent`:
   `"violence detected in gym_east (camera CAM_07) at 17:00 on weekday. Confidence 0.78. Involving 2 people."`
2. Embed this string with `all-MiniLM-L6-v2` → a 384-dimensional vector (~4ms on CPU).
3. Query Qdrant's `dismissed_alerts` collection: ANN search for the nearest vector, filtered to `event_type` match and `ttl_expires` not yet passed.
4. If the cosine similarity of the nearest result exceeds a configurable, per-event-type threshold → suppress.

**Why text embeddings rather than raw numeric feature vectors:** Different event types have completely different feature sets (violence → inter-person distance/velocity; fire → smoke region area/persistence). A fixed-length numeric vector across all event types would require awkward padding and per-type normalization. A natural-language description naturally absorbs any feature set, and a sentence embedding model captures *semantic* similarity regardless of which specific numbers are present.

**Why this needs to be a separate layer from 4A, not merged:** They use entirely different storage systems and answer entirely different questions. 4A asks "has an operator drawn an explicit boundary around this exact situation?" 4B asks "does this look like something an operator has seen and dismissed before, even if not identical?" Conflating them would prevent independently tuning each.

| Event Type  | Similarity Threshold | Rationale |
|-------------|----------------------|-----------|
| Fire        | 0.95 | Almost never auto-suppress — miss cost too high |
| Violence    | 0.88 | Suppress recurring patterns like gym sports |
| Dog Attack  | 0.85 | Repetitive dog behavior in same area is common |
| Trespassing | 0.90 | Authorized-personnel patterns are fairly specific |
| Accident    | 0.93 | True repeats are rare |

---

### Layer 5 — Tier Assignment and Routing

**What it does:** Assigns every surviving alert a severity tier, which determines dashboard prominence, notification behavior, and whether acknowledgment is required.

```
CRITICAL  →  Fire (any confidence above threshold)
              Violence confidence > 0.90

HIGH      →  Violence confidence 0.72–0.90
              Dog Attack confidence > 0.80
              Accident confidence > 0.80

MEDIUM    →  Trespassing
              Dog Attack 0.68–0.80
              Accident 0.70–0.80

LOW       →  Near-threshold events surviving suppression
```

Tier mappings are also stored in Postgres config — not hardcoded.

**Output object — `DashboardAlert`:**

```
alert_id            UUID
tier                "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
event_type          str
camera_id           str
zone_id             str
zone_label          str
raw_confidence      float
effective_conf      float
timestamp           datetime
evidence_frame_ref  str
involved_ids        List[int]
suppression_score   float   — cosine similarity of nearest dismissed alert,
                               even if below suppression threshold (transparency)
```

`suppression_score` is included even on alerts that *aren't* suppressed — it gives the operator a signal like "this is 84% similar to something you dismissed before" without auto-hiding it, surfacing tuning opportunities.

---

### The Feedback Loop

**On "Not Important" (dismiss):**
1. Write a suppression rule to Postgres (`expires_at = now + 24h` by default, or `NULL` if "always ignore" is checked)
2. Build the alert's description string, embed it, write the vector + payload to Qdrant `dismissed_alerts`
3. *(Architecturally correct version: both writes go through the Outbox Pattern — see §4.1)*

**On "Confirm" (genuine event):**
1. Log to the Postgres/TimescaleDB `alert_log` hypertable as confirmed
2. Optionally embed and store in a `confirmed_alerts` Qdrant collection for future use
3. Optionally clear the Redis dedup key if re-alerting is desired for an escalating ongoing incident

**Auto-promotion:** A background job periodically counts dismissals grouped by `(camera_id, zone_id, event_type, hour_band)`. If a combination has been dismissed 5+ times, its suppression rule is automatically promoted to a longer TTL (or permanent), and the dashboard is notified that a new auto-rule was created — keeping operators in the loop rather than silently changing behavior.

---

## 4. Concept Deep Dives

### 4.1 The Dual-Write Problem and the Outbox Pattern

When an operator dismisses an alert, two independent databases must be updated: Postgres (suppression rule) and Qdrant (embedding). These are separate systems with no shared transaction mechanism — if Postgres succeeds and Qdrant fails (e.g., briefly down), the system ends up in an inconsistent state. This is the **dual-write problem**, one of the most fundamental issues in distributed systems, and there is no way to make two independent databases commit atomically without a distributed transaction coordinator (two-phase commit) — which is slow and operationally painful, and essentially never used for workloads like this.

**The standard solution: the Outbox Pattern.** Instead of writing to Postgres and Qdrant directly in two separate calls, the application writes *both* intended operations as rows in an "outbox" table, inside the *same* Postgres transaction as the suppression rule write. Because both writes are part of one Postgres transaction, they are atomic with each other — either both happen or neither does. A separate background worker then reads the outbox table and performs the Qdrant write, marking the row complete on success and retrying on failure. The Qdrant write becomes *eventually consistent* (within a second or two) rather than immediately consistent — which is fine here, since suppression doesn't need to take effect instantaneously.

For a capstone scope, sequential writes with a logged warning on partial failure is an honest, acceptable simplification — but knowing the outbox pattern exists, and being able to say "this is the architecturally correct solution and here's why," is the kind of answer that distinguishes a well-reasoned design from a naive one.

---

### 4.2 In-Memory Key-Value Stores and TTL (Redis)

Redis is a database that lives entirely in RAM rather than on disk. Reading from RAM takes roughly 100 nanoseconds; reading from an SSD takes roughly 100 microseconds — a ~1000x difference. For a check that runs on every single incoming event across potentially 100 cameras at 15fps (1500 events/second), this difference is the difference between a real-time pipeline and one that falls permanently behind.

**TTL (Time To Live)** is a timer attached to a key. `SETEX key 45 value` stores a key that Redis automatically deletes after 45 seconds — no cleanup code required. This is the exact primitive burst deduplication needs: write a key when an incident starts, let it auto-expire when the dedup window ends, and the next alert of that type then passes through naturally.

**Mental model:** a whiteboard in a security room. Writing "CAM_07:violence" with a self-destructing sticky note that disappears in 45 seconds. Any alert arriving while the note is up gets binned as a duplicate; once it's gone, the next alert passes through.

---

### 4.3 Vectors and Embeddings

A **vector** is simply a list of numbers, interpretable as coordinates of a point in space. `[0.2, -0.5, 0.8]` is a point in 3D space; a 384-dimensional vector is a point in 384-dimensional space — impossible to visualize, but mathematically well-defined.

An **embedding** is the vector produced by feeding text (or images, audio, etc.) through a neural network specifically trained to produce them. The defining property: **text with similar meaning produces vectors that are close together**, even if the sentences share almost no words. "A fight broke out near the gym" and "two people were fighting in the sports hall" produce nearly identical vectors despite minimal lexical overlap — the model captures *meaning*, not surface text.

This is exactly why it works for suppression: two false-positive descriptions of the same underlying situation (e.g. basketball practice mistaken for fighting), phrased slightly differently or occurring on different days, still land close together in vector space. A plain text/keyword search in Postgres would miss this; embedding similarity catches it.

---

### 4.4 Cosine Similarity

Cosine similarity measures the **angle** between two vectors, not the distance between them, and returns a value in [-1, 1]: 1 means the vectors point in the same direction (same meaning), 0 means unrelated, -1 means opposite. For suppression: if the cosine similarity between a new alert's embedding and a dismissed alert's embedding exceeds the configured threshold (e.g. 0.88 for violence), the new alert is suppressed as "semantically too similar to something already dismissed."

---

### 4.5 Approximate Nearest Neighbor (ANN) Search and HNSW

With tens of thousands of stored dismissed-alert vectors, finding the *exact* nearest neighbor to a new vector by comparing against every stored vector becomes slow as the collection grows. **ANN (Approximate Nearest Neighbor)** algorithms trade a small, tunable amount of accuracy for dramatic speed gains.

Qdrant implements **HNSW (Hierarchical Navigable Small World)**. Picture the stored vectors as points scattered through high-dimensional space, connected in a **multi-layer graph**: the top layer has very few nodes connected by long-range links spanning the whole space; each layer below is progressively denser, with progressively shorter-range links, until the bottom layer contains every vector connected only to its true close neighbors.

A search starts at the sparse top layer and greedily walks toward whichever neighbor is closest to the query vector. When it can't get any closer at that layer, it drops down to the next denser layer and continues from where it landed. By the bottom layer, the search is already in the right *neighborhood* and only needs to examine a small local cluster — never the whole dataset.

The analogy: navigating to an address by first picking the right continent, then country, then city, then street — never comparing your destination against every street on Earth. "Small world" refers to the graph-theory idea that any two nodes are connected through surprisingly few hops (the "six degrees of separation" principle). The "approximate" part comes from the greedy walk occasionally settling for a near-best rather than the absolute best — Qdrant's `ef_search` parameter trades accuracy for speed, but at the scale of a few thousand dismissed alerts, HNSW will essentially always return the true nearest neighbor.

---

### 4.6 Sentence Transformers and `all-MiniLM-L6-v2`

A transformer encoder processes a sentence token by token (a token ≈ a word or sub-word piece). Each token starts as a context-free embedding from a lookup table — "fight" has the same starting vector whether in "a fight broke out" or "a fight for justice."

The **attention mechanism** adds context: for every token, the model computes a weighted combination of *all other tokens in the sentence*, with weights learned based on relevance. Processing "fight" in "a fight broke out near the gym" pulls in information from "broke," "out," and "gym," shifting the resulting vector toward "physical altercation" rather than "advocacy." This happens across 6 stacked layers (the "L6" in MiniLM), each layer refining the representation further.

To collapse the sequence of per-token vectors into one sentence vector, the model applies **mean pooling** — averaging all token vectors (with normalization).

The reason similar sentences land close together is purely a product of **training**: the model was trained on millions of sentence pairs with a *contrastive objective* — semantically similar pairs are pushed together in vector space, dissimilar pairs are pushed apart, repeated over millions of examples until this geometric property emerges. It is learned, not hand-designed, which is why it generalizes to novel alert descriptions it never saw during training.

**Distillation** ("Mini" in MiniLM): a large "teacher" model's output vectors were used as training targets for a much smaller "student" model, which learns to approximate the teacher's vector space with far fewer parameters. This is why a 22MB model running in ~4ms on CPU can do something that originally required a model ~10x larger.

---

### 4.7 Precision, Recall, and Threshold Selection

Every classifier outputs a confidence score, and a **cutoff (threshold)** decides what counts as a positive detection. Lowering the cutoff catches more true positives (**higher recall**) but lets in more false positives (**lower precision**); raising it does the reverse. No single cutoff maximizes both — this is a mathematical property of any imperfect classifier, not a flaw specific to any model here.

The **ROC curve** plots true-positive rate vs false-positive rate across all thresholds. The **precision-recall curve** is often more informative for rare events (violence is rare relative to all video frames), because ROC curves can look deceptively good when negatives vastly outnumber positives.

This is *why* the base thresholds differ by event type: each represents a different point on this curve, chosen based on the **cost asymmetry of errors**. Fire's low threshold (0.60) accepts more false alarms to avoid ever missing a real fire (false negatives are catastrophic). Trespassing's high threshold (0.78) accepts missing some borderline cases to avoid constant false alerts from authorized foot traffic (false positives cause alert fatigue). In a fully rigorous setting these cutoffs would be chosen empirically from precision-recall curves on validation data; for this project, reasoned defaults plus dashboard-based operator tuning achieve the same end interactively.

---

### 4.8 Hysteresis

Borrowed from control theory: a thermostat doesn't switch off at exactly 70°F and back on at exactly 70°F — it would oscillate rapidly ("chatter") as the temperature hovers near that point. Instead it uses two different thresholds for the two directions (e.g. off at 72°F, on at 68°F).

The Alert Engine has the same risk: if a detection confidence hovers right around 0.72, a naive system alerts → suppresses → alerts → suppresses every frame as the value noisily crosses the line. Two mechanisms apply hysteresis here: requiring **sustained** above-threshold confidence across multiple consecutive windows before the *initial* alert fires, and requiring a confidence **jump of 0.15 or more** before re-alerting on an already-active incident (Layer 3). Both are instances of the same named, well-studied principle — citing it by name signals a general understanding rather than an ad-hoc fix.

---

### 4.9 Bitmask Encoding (`days_mask`)

`days_mask` stores a *set of days* in a single integer using **binary flag encoding**: Mon=1 (`0000001`), Tue=2 (`0000010`), Wed=4, Thu=8, Fri=16, Sat=32, Sun=64. "Weekdays" = 1+2+4+8+16 = 31 (`0011111`).

Membership is checked with **bitwise AND**: `days_mask & (1 << day_of_week) > 0`. AND-ing two binary numbers is non-zero only where both have a 1 in the same position — so this single arithmetic operation tests "is today in this rule's day set?" without a separate join table of (rule_id, day) rows.

---

### 4.10 Idempotency in Event Pipelines

An operation is **idempotent** if running it multiple times produces the same result as running it once. Most message/event delivery systems (including Redis Streams, used elsewhere in the project) guarantee "at-least-once" delivery, not "exactly-once" — **distributed systems fail in ways that cause duplicate messages far more often than lost ones.**

The defense: `alert_id` should be a **deterministic** identifier — derived from a hash of `(camera_id, timestamp, event_type)` rather than a random UUID — so that if the same underlying detection is processed twice (e.g., due to an upstream retry), both attempts produce the *same* `alert_id`. A unique constraint on `alert_id` in the Postgres `alert_log` table then lets downstream consumers silently discard the duplicate. Designing for idempotency from the start means duplicate deliveries are harmless rather than producing duplicate dashboard alerts.

---

### 4.11 Database Indexing (B-trees)

Postgres, by default, indexes columns using **B-trees** — balanced tree structures where each comparison eliminates roughly half the remaining search space, giving O(log n) lookups instead of O(n) full table scans.

For `suppression_rules`, a **composite index** on `(camera_id, event_type)` — the two columns always filtered on first — lets Postgres jump almost directly to relevant rows. With this index, even a table with millions of rows returns results in sub-millisecond time; without it, query time grows linearly with table size — fine in a demo with 100 rows, a real problem after a semester of accumulated data. The fix costs one line: `CREATE INDEX ON suppression_rules (camera_id, event_type);`

---

### 4.12 Vector Quantization (Scaling Beyond This Project)

50,000 vectors of 384 float32 values ≈ 75MB — trivial. Production vector databases handling hundreds of millions of vectors face real storage costs. Qdrant supports **quantization**: compressing each vector's numbers from 32-bit floats to 8-bit integers (scalar quantization) or even single bits (binary quantization), trading a small accuracy loss in distance computation for large storage/speed gains.

Not needed at this project's scale, but it directly answers the natural follow-up question — "does this approach scale to a real multi-campus deployment with millions of historical alerts?" — with a concrete mechanism rather than a hand-wave.

---

## 5. Technology Stack Summary

| Component | Technology | Role | Why This, Not the Alternative |
|---|---|---|---|
| Burst dedup | **Redis** (TTL keys) | Sub-ms duplicate suppression for active incidents | Postgres write+read per event becomes a bottleneck at scale; Redis is ~1000x faster (RAM vs disk) and has native TTL |
| Exact suppression rules, config, alert log | **PostgreSQL / TimescaleDB** | Structured rules, thresholds, audit trail | Transactional guarantees; TimescaleDB = Postgres + time-series extensions, so `alert_log` can be a hypertable with zero added services |
| Semantic suppression | **Qdrant** | ANN similarity search over dismissed-alert embeddings | Purpose-built indexing (HNSW) + payload filtering; `pgvector` is an acceptable simplification at small scale, Qdrant is the architecturally correct, scalable choice |
| Embedding model | **all-MiniLM-L6-v2** | Text → 384-dim semantic vector | 22MB, <5ms on CPU, no external API, trained specifically for semantic similarity |
| Inter-module transport | Redis Pub/Sub or stream | Decouple Alert Engine from dashboard | Matches the rest of the system's existing Redis Streams usage |

---

## 6. Integration Contract (for Teammates)

Your module is a black box that consumes one object type and produces one object type. Anyone integrating with you only needs to know these two shapes — everything inside Layers 1–5 is your implementation detail.

**Input — from upstream classifiers (via Aastha's tracking/feature pipeline):**

```python
class RawDetectionEvent:
    pipeline: str            # "violence" | "fire" | "dog_attack" | "trespassing" | "accident"
    raw_confidence: float
    camera_id: str
    zone_id: str
    zone_label: str
    timestamp: datetime
    frame_ref: str
    involved_ids: list[int]
    pipeline_features: dict
```

**Output — to dashboard (Vaibhav's module):**

```python
class DashboardAlert:
    alert_id: str             # deterministic UUID
    tier: str                  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    event_type: str
    camera_id: str
    zone_id: str
    zone_label: str
    raw_confidence: float
    effective_conf: float
    timestamp: datetime
    evidence_frame_ref: str
    involved_ids: list[int]
    suppression_score: float
```

**Feedback — from dashboard, back into your module:**

```python
class OperatorFeedback:
    alert_id: str
    action: str               # "dismiss" | "confirm"
    permanent: bool           # only relevant if action == "dismiss"
```

Open items to confirm with the team before implementation begins (carried over, unchanged from earlier discussion): the exact `pipeline_features` schema per event type, the dashboard's consumption interface (websocket vs Redis stream), confirmation that suppression config/rules live in the shared Postgres/TimescaleDB instance, and Qdrant deployment via Docker on the shared server.

---

## 7. Where to Start: Implementation Roadmap

Build this **bottom-up, layer by layer, each independently testable before wiring them together.** Suggested module structure:

```
alert_engine/
├── config/
│   └── thresholds.py        # loads/caches threshold + tier config from Postgres
├── models/
│   └── schemas.py            # AlertEvent, DashboardAlert, RawDetectionEvent, OperatorFeedback
├── layers/
│   ├── threshold_gate.py      # Layer 1
│   ├── enrichment.py          # Layer 2
│   ├── dedup.py                # Layer 3 (Redis)
│   ├── suppression/
│   │   ├── exact_rules.py      # Layer 4A (Postgres)
│   │   └── semantic.py         # Layer 4B (Qdrant + embedding)
│   └── tiering.py              # Layer 5
├── feedback/
│   └── handler.py              # processes OperatorFeedback, writes suppression rule + embedding
├── pipeline.py                  # orchestrates Layers 1-5 in sequence
└── tests/
    └── (one test module per layer, with mocked dependencies)
```

**Recommended build order:**

1. **Schemas first** (`models/schemas.py`) — define `RawDetectionEvent`, `AlertEvent`, `DashboardAlert`, `OperatorFeedback` exactly as in §6. Everything else depends on these being stable, and this is also the artifact you can hand to teammates immediately for parallel integration work.

2. **Layer 1 (threshold gate)** — start with hardcoded config dict matching §3's tables, get the time/zone multiplier math working and unit-tested. Swap in real Postgres-backed config later without changing the interface.

3. **Layer 2 (enrichment)** — trivial in isolation; mostly a constructor. Write it once Layer 1's output shape is settled.

4. **Layer 3 (Redis dedup)** — stand up a local Redis instance (Docker), implement the TTL key logic and the hysteresis escalation check. This is fully testable in isolation with `fakeredis` or a real local Redis.

5. **Layer 4A (Postgres exact rules)** — stand up local Postgres/TimescaleDB (Docker), create the `suppression_rules` table with the index, implement and test the lookup query.

6. **Layer 4B (Qdrant semantic suppression)** — stand up Qdrant (Docker), implement the description-string builder, integrate `sentence-transformers` for embedding, implement the ANN query with payload filtering. This is the most novel piece — build it last among the suppression components so the simpler pieces are already solid.

7. **Layer 5 (tiering)** — straightforward rule mapping once everything above produces a surviving `AlertEvent`.

8. **`pipeline.py`** — wire Layers 1–5 in sequence, end to end, with a mocked `RawDetectionEvent` feed.

9. **Feedback handler** — implement the dismiss/confirm write-back. Start with simple sequential writes (Postgres then Qdrant) with logging on partial failure; note the Outbox Pattern (§4.1) as the documented future improvement.

10. **Integration testing** — once your teammates' modules produce real `RawDetectionEvent` objects and the dashboard can consume real `DashboardAlert` objects, replace your mocks with the real connections one at a time.

This order ensures that at every step you have a working, testable component — and the schema-first approach means your teammates can build against your interface long before your internal layers are finished.
