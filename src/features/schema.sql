-- Star Schema DDL for Consumer Telemetry Data Warehouse

CREATE TABLE IF NOT EXISTS fact_events (
    event_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    app_name VARCHAR NOT NULL,
    canonical_product_id VARCHAR,
    action VARCHAR NOT NULL,
    -- DECIMAL(18,2) rather than (10,2): a real clickstream overflowed the narrower
    -- column. RetailRocket's obfuscated prices reach 1.25e8 against a (10,2) ceiling
    -- of 1e8. Synthetic values peak near 1.3e5, so widening changes nothing already
    -- stored and only removes a limit that real data does exceed.
    price DECIMAL(18,2),
    currency VARCHAR,
    session_id VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_products (
    canonical_product_id VARCHAR PRIMARY KEY,
    brand VARCHAR,
    product_title VARCHAR,
    category_l1 VARCHAR,
    category_l2 VARCHAR,
    avg_price DECIMAL(18,2)
);

CREATE TABLE IF NOT EXISTS dim_users (
    user_id VARCHAR PRIMARY KEY,
    first_seen TIMESTAMP,
    primary_app VARCHAR,
    total_events INTEGER
);


CREATE TABLE IF NOT EXISTS fact_cross_app_comparisons (
    comparison_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR,
    canonical_product_id VARCHAR NOT NULL,
    primary_app VARCHAR NOT NULL,
    compared_app VARCHAR NOT NULL,
    first_viewed TIMESTAMP NOT NULL,
    last_viewed TIMESTAMP NOT NULL,
    duration_minutes DECIMAL(10,2) NOT NULL
);


-- Indexes for accelerated analytical filtering and session joins
CREATE INDEX IF NOT EXISTS idx_fact_user_ts ON fact_events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_product ON fact_events(canonical_product_id);
CREATE INDEX IF NOT EXISTS idx_fact_session ON fact_events(session_id);
CREATE INDEX IF NOT EXISTS idx_cross_app_user ON fact_cross_app_comparisons(user_id);
