-- ============================================================
-- ZaiGuard Alert Engine — Seed Data
-- File: db/init/02_seed_data.sql
-- Runs after 01_schema.sql (alphabetical order).
-- Populates config tables with initial values for all pipelines.
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- threshold_config — per-pipeline base thresholds
-- ─────────────────────────────────────────────────────────────

-- Existing pipelines (from architecture doc)
INSERT INTO threshold_config (pipeline, base_threshold, dedup_ttl_seconds, escalation_delta)
VALUES
    ('fire',         0.60, 120, 0.15),
    ('violence',     0.72,  45, 0.15),
    ('dog_attack',   0.68,  60, 0.15),
    ('trespassing',  0.78,  90, 0.15),
    ('accident',     0.70,  60, 0.15),
    ('unauth_access', 0.50,  30, 0.15)
ON CONFLICT (pipeline) DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- time_multiplier_config — sensitivity by time of day
-- ─────────────────────────────────────────────────────────────
INSERT INTO time_multiplier_config (hour_start, hour_end, multiplier, label)
VALUES
    (0,  5,  0.70, 'late_night'),
    (6,  8,  0.85, 'early_morning'),
    (9, 17,  1.00, 'business_hours'),
    (18, 21, 0.90, 'evening'),
    (22, 23, 0.75, 'night')
ON CONFLICT DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- zone_config — default zone + example zones
-- ─────────────────────────────────────────────────────────────
INSERT INTO zone_config (zone_id, zone_label, risk_multiplier)
VALUES
    ('default', 'Default Zone', 1.0)
ON CONFLICT (zone_id) DO NOTHING;

-- ─────────────────────────────────────────────────────────────
-- tier_config — severity tier mappings per pipeline
-- Rows are evaluated highest min_confidence first.
-- ─────────────────────────────────────────────────────────────

-- Fire: any confidence above threshold is CRITICAL
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('fire', 0.60, 'CRITICAL');

-- Violence
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('violence', 0.90, 'CRITICAL'),
    ('violence', 0.72, 'HIGH'),
    ('violence', 0.00, 'MEDIUM');

-- Dog attack
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('dog_attack', 0.80, 'HIGH'),
    ('dog_attack', 0.68, 'MEDIUM'),
    ('dog_attack', 0.00, 'LOW');

-- Trespassing
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('trespassing', 0.90, 'HIGH'),
    ('trespassing', 0.78, 'MEDIUM'),
    ('trespassing', 0.00, 'LOW');

-- Accident
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('accident', 0.80, 'HIGH'),
    ('accident', 0.70, 'MEDIUM'),
    ('accident', 0.00, 'LOW');

-- Unauthorized access (from Prototype)
INSERT INTO tier_config (pipeline, min_confidence, tier) VALUES
    ('unauth_access', 0.90, 'CRITICAL'),
    ('unauth_access', 0.75, 'HIGH'),
    ('unauth_access', 0.60, 'MEDIUM'),
    ('unauth_access', 0.00, 'LOW');

-- ─────────────────────────────────────────────────────────────
-- suppression_similarity_config — per-pipeline Qdrant thresholds
-- ─────────────────────────────────────────────────────────────
INSERT INTO suppression_similarity_config (pipeline, similarity_threshold)
VALUES
    ('fire',          0.95),
    ('violence',      0.88),
    ('dog_attack',    0.85),
    ('trespassing',   0.90),
    ('accident',      0.93),
    ('unauth_access', 0.90)
ON CONFLICT (pipeline) DO NOTHING;
