CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS prediction_events (
    prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    request_id text,
    model_name text NOT NULL,
    model_version text,
    model_alias text,
    features jsonb NOT NULL,
    predicted_class text NOT NULL,
    probabilities jsonb,
    latency_ms double precision,
    actual_class text,
    feedback_at timestamptz,
    feedback_comment text,
    is_correct boolean
);

CREATE INDEX IF NOT EXISTS idx_prediction_events_created_at
    ON prediction_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_events_feedback_at
    ON prediction_events (feedback_at DESC)
    WHERE actual_class IS NOT NULL;

CREATE TABLE IF NOT EXISTS monitoring_runs (
    monitoring_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    sample_size integer NOT NULL DEFAULT 0,
    drift_ratio double precision,
    drift_detected boolean NOT NULL DEFAULT false,
    labeled_sample_size integer NOT NULL DEFAULT 0,
    rolling_accuracy double precision,
    rolling_f1_weighted double precision,
    retraining_required boolean NOT NULL DEFAULT false,
    reason text,
    report jsonb NOT NULL DEFAULT '{}'::jsonb
);
