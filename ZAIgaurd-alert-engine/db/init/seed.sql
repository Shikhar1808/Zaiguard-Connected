-- ============================================================
-- ZaiGuard Alert Engine — Default Configuration Seed Data
-- File: db/init/02_seed.sql
-- Runs after 01_schema.sql on first container startup.
-- All values here match the architecture document (§3).
-- Every value is tunable from the dashboard — nothing hardcoded
-- in application code.
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- Per-pipeline thresholds and dedup windows
-- ─────────────────────────────────────────────────────────────
INSERT INTO threshold_config (pipeline, base_threshold, dedup_ttl_seconds, escalation_delta)
VALUES
    ('fire',          0.60, 120, 0.15),  -- fire stays active longer; 120s TTL
    ('violence',      0.72,  45, 0.15),
    ('dog_attack',    0.68,  45, 0.15),
    ('trespassing',   0.78,  60, 0.15),
    ('accident',      0.70,  60, 0.15),
    ('unauth_access', 0.50,  30, 0.15)   -- Prototype: fast-fire, short dedup window
ON CONFLICT (pipeline) DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- Time-of-day multipliers
-- Nighttime (22:00–05:59 UTC): more sensitive (lower effective threshold)
-- Peak hours (08:00–18:59 UTC): less sensitive (higher effective threshold)
-- Off-peak (06:00–07:59, 19:00–21:59): neutral
-- ─────────────────────────────────────────────────────────────
INSERT INTO time_multiplier_config (hour_start, hour_end, multiplier, label)
VALUES
    (22, 23, 0.85, 'nighttime_late'),
    ( 0,  5, 0.85, 'nighttime_early'),
    ( 6,  7, 1.00, 'off_peak'),
    ( 8, 18, 1.10, 'peak_hours'),
    (19, 21, 1.00, 'off_peak_evening')
ON CONFLICT DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- Default zone configs
-- Zone IDs here are placeholders — real ones come from the
-- camera/zone registration step (owned by another teammate).
-- risk_multiplier > 1.0 = more sensitive (restricted areas)
-- risk_multiplier < 1.0 = less sensitive (busy public areas)
-- ─────────────────────────────────────────────────────────────
INSERT INTO zone_config (zone_id, zone_label, risk_multiplier)
VALUES
    ('default',        'Unknown Zone',         1.00),
    ('restricted_high','Restricted Area',       0.85),  -- more sensitive
    ('public_high',    'High-Traffic Public',   1.15),  -- less sensitive (noisy)
    ('public_low',     'Low-Traffic Public',    1.00),
    ('parking',        'Parking Area',          0.95),
    ('entrance',       'Campus Entrance',       0.90)
ON CONFLICT (zone_id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- Tier assignment rules
-- Evaluated per pipeline, highest min_confidence row first.
-- Application picks the first row where effective_conf >= min_confidence.
-- ─────────────────────────────────────────────────────────────
INSERT INTO tier_config (pipeline, min_confidence, tier)
VALUES
    -- Fire: always CRITICAL above base threshold
    ('fire',        0.60, 'CRITICAL'),

    -- Violence
    ('violence',    0.90, 'CRITICAL'),
    ('violence',    0.72, 'HIGH'),

    -- Dog attack
    ('dog_attack',  0.80, 'HIGH'),
    ('dog_attack',  0.68, 'MEDIUM'),

    -- Trespassing: always MEDIUM (high false-positive rate)
    ('trespassing', 0.78, 'MEDIUM'),

    -- Accident
    ('accident',    0.80, 'HIGH'),
    ('accident',    0.70, 'MEDIUM'),

    -- Unauthorized access (from Prototype classifier)
    ('unauth_access', 0.90, 'CRITICAL'),
    ('unauth_access', 0.75, 'HIGH'),
    ('unauth_access', 0.60, 'MEDIUM'),
    ('unauth_access', 0.00, 'LOW')
ON CONFLICT DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- Semantic similarity thresholds for Qdrant suppression (Layer 4B)
-- Conservative defaults — start tight, operators loosen over time
-- ─────────────────────────────────────────────────────────────
INSERT INTO suppression_similarity_config (pipeline, similarity_threshold)
VALUES
    ('fire',          0.95),  -- almost never auto-suppress fire
    ('violence',      0.88),
    ('dog_attack',    0.85),
    ('trespassing',   0.90),
    ('accident',      0.93),
    ('unauth_access', 0.90)
ON CONFLICT (pipeline) DO NOTHING;