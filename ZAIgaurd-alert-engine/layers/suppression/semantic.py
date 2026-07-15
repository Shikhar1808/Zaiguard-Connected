"""
ZaiGuard Alert Engine — Layer 4B: Qdrant Semantic Suppression
=============================================================
Compares incoming alerts against previously dismissed alerts in Qdrant
using semantic vector search (Approximate Nearest Neighbors — ANN).

POSITION IN PIPELINE
--------------------
Layer 4B runs AFTER exact rule suppression (Layer 4A) and BEFORE
tiering (Layer 5). While Layer 4A checks for explicit operator rules,
Layer 4B detects soft recurring patterns — e.g. "janitor cleaning gym
after hours" or "sunlight glare triggering fire detection at entrance"
— that an operator has repeatedly dismissed, even if slight variations
exist in confidence, exact time, or feature values.

THREE SUB-PIECES
----------------
1. Description-String Builder (build_alert_description):
   Converts structured event fields into a clean natural-language
   summary. Sentence transformers capture similarity best when
   domain attributes are presented coherently and deterministically.

2. Embedding Manager (SemanticEmbedder):
   Lazy-loads all-MiniLM-L6-v2 via sentence-transformers. Output
   vectors (384-d) are L2-normalized so cosine similarity equals
   the dot product — consistent with Qdrant's Distance.COSINE metric.

3. Qdrant ANN Search (run_semantic_suppression_check):
   Queries the dismissed_alerts collection with two filters:
     a) pipeline must match (never let a dismissed fire alert
        suppress a genuine violence alert)
     b) TTL not expired — non-permanent dismissals are time-limited
        (default 24h) so stale operator feedback doesn't suppress
        forever unless explicitly marked permanent

TTL DESIGN
----------
Every dismissed alert stored in Qdrant carries two payload fields:
  is_permanent: bool  — True if operator chose "always ignore"
  ttl_expires:  str   — ISO datetime after which this dismissal
                        expires, or None if is_permanent=True

The ANN search uses a Qdrant `should` filter to match vectors where
EITHER is_permanent==True OR ttl_expires >= now. This mirrors the
expiry logic in Layer 4A's suppression_rules table and ensures
non-permanent dismissals have a consistent 24h lifetime across both
suppression layers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from config.settings import settings
from config.thresholds import config_loader
from models.schemas import AlertEvent, RawDetectionEvent

logger = logging.getLogger(__name__)

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

# Default TTL for non-permanent dismissed alerts — matches Layer 4A
DEFAULT_DISMISSAL_TTL_HOURS = 24


# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SemanticRuleResult:
    """
    Return value of run_semantic_suppression_check().

    suppressed:
        True  → drop event, strongly matches a dismissed pattern
        False → continue to Layer 5

    nearest_score:
        Cosine similarity (0.0–1.0) of the closest dismissed alert.
        Attached to DashboardAlert even when not suppressed — gives
        operators context like "84% similar to something you dismissed"
        and surfaces tuning opportunities for the dashboard.

    nearest_alert_id:
        The alert_id of the matched dismissed alert in Qdrant, or None.
    """
    suppressed: bool
    nearest_score: float = 0.0
    nearest_alert_id: str | None = None


# ─────────────────────────────────────────────────────────────
# Sub-piece 1: Description-String Builder
# ─────────────────────────────────────────────────────────────

def build_alert_description(event: AlertEvent | RawDetectionEvent) -> str:
    """
    Constructs a deterministic natural-language representation of an event.

    WHY TEXT INSTEAD OF A RAW NUMERIC VECTOR
    -----------------------------------------
    Different pipelines have completely different feature sets —
    violence gives inter-person distance and velocity; fire gives
    smoke area and persistence. A fixed-length numeric vector would
    require pipeline-specific normalization and awkward padding.
    Text handles any pipeline naturally, and the sentence transformer
    captures semantic similarity regardless of which specific numbers
    are present.

    WHY SORTED FEATURES
    --------------------
    dict insertion order is deterministic in Python 3.7+, but
    pipeline_features could arrive with keys in any order depending
    on which upstream module built it. Sorting by key ensures two
    events with identical feature values always produce the same
    description string and therefore the same embedding.

    Example output:
        "Pipeline: violence | Camera: CAM_07 | Zone: Gym East |
         Time: 17:00 on Monday | Features: inter_person_distance=0.5,
         relative_velocity=2.1"
    """
    pipeline_str = (
        event.pipeline.value
        if hasattr(event.pipeline, "value")
        else str(event.pipeline)
    )

    # AlertEvent has pre-computed hour/day; RawDetectionEvent has timestamp
    if hasattr(event, "hour_of_day"):
        hour = event.hour_of_day
        day_idx = event.day_of_week
    else:
        hour = event.timestamp.hour
        day_idx = event.timestamp.weekday()

    day_name = DAYS_OF_WEEK[day_idx] if 0 <= day_idx <= 6 else "Unknown"

    # Resolve pipeline_features from AlertEvent or RawDetectionEvent
    feats = getattr(event, "pipeline_features", None)
    if feats is None and hasattr(event, "source_event"):
        feats = getattr(event.source_event, "pipeline_features", {})
    feats = feats or {}

    feat_str = (
        ", ".join(f"{k}={v}" for k, v in sorted(feats.items()))
        if feats
        else "none"
    )

    return (
        f"Pipeline: {pipeline_str} | "
        f"Camera: {event.camera_id} | "
        f"Zone: {event.zone_label} | "
        f"Time: {hour:02d}:00 on {day_name} | "
        f"Features: {feat_str}"
    )


# ─────────────────────────────────────────────────────────────
# Sub-piece 2: Embedding Manager
# ─────────────────────────────────────────────────────────────

class SemanticEmbedder:
    """
    Singleton wrapper around SentenceTransformer.

    Lazy-loads the model on first encode() call so that application
    startup and test collection stay fast — the model is only loaded
    when actually needed. Once loaded, it is cached for the lifetime
    of the process.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.embedding_model_name
        self._cache_dir = cache_dir or settings.embedding_cache_dir
        self._model = None

    def ensure_loaded(self) -> None:
        if self._model is None:
            logger.info(
                "Loading semantic embedding model '%s'...", self._model_name
            )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_dir,
            )
            logger.info("Semantic embedding model loaded.")

    def encode(self, text: str) -> list[float]:
        """
        Encodes text into a 384-dimensional L2-normalized vector.

        normalize_embeddings=True ensures cosine similarity equals
        the dot product, consistent with Qdrant's Distance.COSINE.
        """
        self.ensure_loaded()
        assert self._model is not None
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


# Module-level singleton — one model instance for the whole application
embedder = SemanticEmbedder()


# ─────────────────────────────────────────────────────────────
# Sub-piece 3: Qdrant Collection Setup
# ─────────────────────────────────────────────────────────────

def _to_point_id(alert_id: str) -> str:
    """
    Converts our 32-char hex alert_id into a hyphenated UUID string
    as required by Qdrant point specifications.

    Falls back to uuid5 for non-hex IDs (e.g. test alert IDs).
    """
    try:
        return str(uuid.UUID(alert_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, alert_id))


async def ensure_qdrant_collections(client: AsyncQdrantClient) -> None:
    """
    Creates the dismissed_alerts collection and its pipeline payload
    index if they don't already exist.

    Called once during FastAPI startup (main.py lifespan, Step 9).
    Safe to call multiple times — existence checks prevent errors.
    """
    try:
        collections_resp = await client.get_collections()
        existing = {c.name for c in collections_resp.collections}

        col = settings.qdrant_collection_dismissed
        if col not in existing:
            logger.info("Creating Qdrant collection '%s'...", col)
            await client.create_collection(
                collection_name=col,
                vectors_config=qmodels.VectorParams(
                    size=384,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Index pipeline for fast payload filtering
            await client.create_payload_index(
                collection_name=col,
                field_name="pipeline",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            # Index is_permanent for fast TTL filter
            await client.create_payload_index(
                collection_name=col,
                field_name="is_permanent",
                field_schema=qmodels.PayloadSchemaType.BOOL,
            )
            # Index ttl_expires for fast datetime range filtering
            await client.create_payload_index(
                collection_name=col,
                field_name="ttl_expires",
                field_schema=qmodels.PayloadSchemaType.DATETIME,
            )
            logger.info("Qdrant collection '%s' initialized.", col)

    except Exception as e:
        logger.error(
            "Failed to ensure Qdrant collections: %s", e, exc_info=True
        )


# ─────────────────────────────────────────────────────────────
# Sub-piece 3 continued: ANN Search
# ─────────────────────────────────────────────────────────────

async def run_semantic_suppression_check(
    event: AlertEvent,
    client: AsyncQdrantClient,
    custom_embedder: SemanticEmbedder | None = None,
) -> SemanticRuleResult:
    """
    Layer 4B: queries Qdrant for dismissed alerts semantically
    similar to this event.

    TWO FILTERS APPLIED
    --------------------
    1. pipeline == event.pipeline.value
       Prevents a dismissed fire alert from ever suppressing a
       genuine violence alert, even if their descriptions overlap.

    2. is_permanent == True  OR  ttl_expires >= now (ISO string)
       Expired non-permanent dismissals are excluded. A dismissed
       basketball practice from 6 months ago should not suppress
       a genuine fight today unless the operator explicitly marked
       it as permanent.

    FAIL-OPEN
    ---------
    Any exception during the Qdrant call returns suppressed=False.
    The failure direction must always be toward showing alerts, never
    silently hiding them because the vector store had a hiccup.

    Parameters
    ----------
    event:           AlertEvent from Layer 2
    client:          AsyncQdrantClient (injected for testability)
    custom_embedder: Optional mock/test embedder (defaults to singleton)
    """
    active_embedder = custom_embedder or embedder
    pipeline_str = event.pipeline.value
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        description = build_alert_description(event)
        vector = active_embedder.encode(description)

        results = await client.search(
            collection_name=settings.qdrant_collection_dismissed,
            query_vector=vector,
            query_filter=qmodels.Filter(
                # ALL of these must be true:
                must=[
                    # 1. Pipeline must match exactly
                    qmodels.FieldCondition(
                        key="pipeline",
                        match=qmodels.MatchValue(value=pipeline_str),
                    ),
                    # 2. TTL: either permanent OR not yet expired
                    qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(
                                key="is_permanent",
                                match=qmodels.MatchValue(value=True),
                            ),
                            qmodels.FieldCondition(
                                key="ttl_expires",
                                range=qmodels.DatetimeRange(gte=now_iso),
                            ),
                        ]
                    ),
                ]
            ),
            limit=1,
        )

        if not results:
            logger.debug(
                "suppression.semantic.no_history",
                extra={
                    "alert_id": event.alert_id,
                    "pipeline": pipeline_str,
                },
            )
            return SemanticRuleResult(suppressed=False, nearest_score=0.0)

        top = results[0]
        score = float(top.score)
        nearest_id = (
            str(top.payload.get("alert_id", top.id))
            if top.payload
            else str(top.id)
        )

        threshold = config_loader.get_similarity_threshold(pipeline_str)

        if score >= threshold:
            logger.info(
                "suppression.semantic.matched",
                extra={
                    "alert_id":              event.alert_id,
                    "pipeline":              pipeline_str,
                    "score":                 score,
                    "threshold":             threshold,
                    "nearest_dismissed_id":  nearest_id,
                },
            )
            return SemanticRuleResult(
                suppressed=True,
                nearest_score=score,
                nearest_alert_id=nearest_id,
            )

        logger.debug(
            "suppression.semantic.below_threshold",
            extra={
                "alert_id":  event.alert_id,
                "pipeline":  pipeline_str,
                "score":     score,
                "threshold": threshold,
            },
        )
        return SemanticRuleResult(
            suppressed=False,
            nearest_score=score,
            nearest_alert_id=nearest_id,
        )

    except Exception as e:
        logger.error(
            "Semantic suppression check failed for alert %s: %s",
            event.alert_id, e, exc_info=True,
        )
        # Fail open — never suppress on error
        return SemanticRuleResult(suppressed=False, nearest_score=0.0)


# ─────────────────────────────────────────────────────────────
# Feedback write path — called by feedback/handler.py (Step 10)
# ─────────────────────────────────────────────────────────────

async def store_dismissed_alert(
    client: AsyncQdrantClient,
    alert_id: str,
    event: AlertEvent | RawDetectionEvent,
    permanent: bool = False,
    custom_embedder: SemanticEmbedder | None = None,
) -> None:
    """
    Embeds and stores a dismissed alert vector in Qdrant.

    Called by feedback/handler.py (Step 10) when an operator
    dismisses an alert. The permanent flag comes from OperatorFeedback:
      - permanent=False → ttl_expires = now + 24h, is_permanent=False
      - permanent=True  → ttl_expires = None,      is_permanent=True

    The TTL payload fields are what run_semantic_suppression_check()
    filters on — they must be written consistently here for the
    filter to work correctly.

    Parameters
    ----------
    client:          AsyncQdrantClient
    alert_id:        The DashboardAlert's alert_id being dismissed
    event:           The original AlertEvent (or RawDetectionEvent)
    permanent:       Whether this suppression should last forever
    custom_embedder: Optional mock embedder for testing
    """
    active_embedder = custom_embedder or embedder
    pipeline_str = (
        event.pipeline.value
        if hasattr(event.pipeline, "value")
        else str(event.pipeline)
    )

    description = build_alert_description(event)
    vector = active_embedder.encode(description)
    point_id = _to_point_id(alert_id)

    # TTL fields — must align with the filter in run_semantic_suppression_check
    if permanent:
        ttl_expires = None
        is_permanent = True
    else:
        ttl_expires = (
            datetime.now(timezone.utc) + timedelta(hours=DEFAULT_DISMISSAL_TTL_HOURS)
        ).isoformat()
        is_permanent = False

    payload = {
        "alert_id":    alert_id,
        "pipeline":    pipeline_str,
        "camera_id":   event.camera_id,
        "zone_id":     event.zone_id,
        "zone_label":  event.zone_label,
        "description": description,
        "timestamp":   (
            event.timestamp.isoformat()
            if hasattr(event.timestamp, "isoformat")
            else str(event.timestamp)
        ),
        "is_permanent": is_permanent,
        "ttl_expires":  ttl_expires,
    }

    await client.upsert(
        collection_name=settings.qdrant_collection_dismissed,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )

    logger.info(
        "suppression.semantic.stored",
        extra={
            "alert_id":    alert_id,
            "point_id":    point_id,
            "pipeline":    pipeline_str,
            "permanent":   is_permanent,
            "ttl_expires": ttl_expires or "permanent",
        },
    )