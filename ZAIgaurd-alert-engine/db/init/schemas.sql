-- ============================================================
-- ZaiGuard Alert Engine — Database Schema
-- File: db/init/01_schema.sql
-- Runs automatically on first container startup.
-- TimescaleDB extension is pre-installed in the image.
-- ============================================================

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ─────────────────────────────────────────────────────────────
-- threshold_config
-- Stores per-pipeline base thresholds and TTL windows for burst
-- dedup. All values editable from the dashboard without code
-- changes. Loaded at startup and cached in memory.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS threshold_config (
    pipeline            TEXT PRIMARY KEY,
    base_threshold      FLOAT NOT NULL,
    dedup_ttl_seconds   INT   NOT NULL,     -- Layer 3: Redis TTL window
    escalation_delta    FLOAT NOT NULL      -- Layer 3: confidence jump to re-alert
                                            --          on an active incident
);

-- ─────────────────────────────────────────────────────────────
-- time_multiplier_config
-- hour_start / hour_end are inclusive. All hours in UTC.
-- A row covering "nighttime" (22:00-06:00) can be represented
-- as two rows (22-23 and 0-6) or handled in application logic.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS time_multiplier_config (
    id              SERIAL PRIMARY KEY,
    hour_start      INT   NOT NULL,
    hour_end        INT   NOT NULL,
    multiplier      FLOAT NOT NULL,
    label           TEXT,                   -- human-readable, e.g. "nighttime"
    CONSTRAINT valid_hours CHECK (hour_start >= 0 AND hour_end <= 23)
);

-- ─────────────────────────────────────────────────────────────
-- zone_config
-- Maps zone_id → risk multiplier and human-readable label.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS zone_config (
    zone_id         TEXT  PRIMARY KEY,
    zone_label      TEXT  NOT NULL,
    risk_multiplier FLOAT NOT NULL DEFAULT 1.0
);

-- ─────────────────────────────────────────────────────────────
-- tier_config
-- Defines which pipeline + confidence range maps to which tier.
-- Multiple rows per pipeline, evaluated in order (highest min
-- threshold first). First matching row wins.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tier_config (
    id              SERIAL PRIMARY KEY,
    pipeline        TEXT  NOT NULL,
    min_confidence  FLOAT NOT NULL,         -- inclusive lower bound
    tier            TEXT  NOT NULL          -- "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
);

-- ─────────────────────────────────────────────────────────────
-- suppression_similarity_config
-- Per-pipeline similarity threshold for Qdrant ANN suppression.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppression_similarity_config (
    pipeline            TEXT  PRIMARY KEY,
    similarity_threshold FLOAT NOT NULL
);

-- ─────────────────────────────────────────────────────────────
-- suppression_rules
-- Explicit operator-defined suppression rules (Layer 4A).
-- days_mask: bitmask Mon=1 Tue=2 Wed=4 Thu=8 Fri=16 Sat=32 Sun=64
-- expires_at NULL = permanent rule
-- source: "manual" (operator-created) | "auto_promoted" (system)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppression_rules (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id       TEXT    NOT NULL,
    zone_id         TEXT,
    pipeline        TEXT    NOT NULL,
    hour_start      INT,
    hour_end        INT,
    days_mask       INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    source          TEXT    NOT NULL DEFAULT 'manual',
    CONSTRAINT valid_source CHECK (source IN ('manual', 'auto_promoted'))
);

-- Composite index on the columns we always filter first
-- Makes Layer 4A lookup sub-millisecond even at millions of rows
CREATE INDEX IF NOT EXISTS idx_suppression_camera_pipeline
    ON suppression_rules (camera_id, pipeline);

-- ─────────────────────────────────────────────────────────────
-- alert_log
-- Permanent audit trail of every alert that reached the dashboard.
-- Created as a TimescaleDB hypertable (partitioned by timestamp)
-- for efficient time-range queries from the dashboard analytics.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alert_log (
    alert_id            TEXT        NOT NULL,
    pipeline            TEXT        NOT NULL,
    tier                TEXT        NOT NULL,
    camera_id           TEXT        NOT NULL,
    zone_id             TEXT        NOT NULL,
    zone_label          TEXT        NOT NULL,
    raw_confidence      FLOAT       NOT NULL,
    effective_conf      FLOAT       NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    evidence_frame_ref  TEXT,
    involved_ids        INT[],
    suppression_score   FLOAT,
    operator_action     TEXT,           -- "confirmed" | "dismissed" | NULL (pending)
    action_at           TIMESTAMPTZ,
    PRIMARY KEY (alert_id, timestamp)
);

-- Convert alert_log to a TimescaleDB hypertable
-- partitioned by timestamp, 1 week per chunk
SELECT create_hypertable(
    'alert_log',
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────────────────────
-- outbox
-- Stores pending cross-database writes (Outbox Pattern, §4.1).
-- A background worker reads this and performs Qdrant writes,
-- retrying on failure. Ensures consistency between Postgres
-- suppression rule writes and Qdrant embedding writes.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outbox (
    id              BIGSERIAL   PRIMARY KEY,
    event_type      TEXT        NOT NULL,   -- "dismissed_alert_embedding"
    payload         JSONB       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ,
    failed_attempts INT         NOT NULL DEFAULT 0,
    last_error      TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_unprocessed
    ON outbox (created_at)
    WHERE processed_at IS NULL;