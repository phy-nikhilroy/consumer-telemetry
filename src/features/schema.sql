-- Star Schema DDL for Consumer Telemetry Data Warehouse, schema version 2.
--
-- This file creates a fresh warehouse at the current version. An existing warehouse at an
-- older version is upgraded by src/features/migrations.py, which runs this file (creating
-- any missing table) and then adds the columns that older versions lack.

CREATE TABLE IF NOT EXISTS schema_info (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL);

-- The screen text exactly as captured, one row per event. fact_events, the price
-- observations, and the product dimension are all derived from it and can be rebuilt.
CREATE TABLE IF NOT EXISTS raw_events (
    event_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    app_name VARCHAR NOT NULL,
    raw_text VARCHAR NOT NULL,
    extractor_version VARCHAR NOT NULL
);

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
    session_id VARCHAR,
    -- Added in schema version 2. All nullable: a screen may carry no MRP, discount, or
    -- rating, and a v1 warehouse has none of them. price_inr is price converted at
    -- USD_TO_INR_RATE, so aggregates never mix currencies.
    mrp DECIMAL(18,2),
    discount_pct INTEGER,
    rating DOUBLE,
    extraction_confidence DOUBLE,
    price_inr DECIMAL(18,2)
);

-- Every priced product seen on a screen, the focal listing and the context listings
-- beside it. is_focal marks the listing the event's action applies to.
CREATE TABLE IF NOT EXISTS fact_price_observations (
    observation_id VARCHAR PRIMARY KEY,
    source_event_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    app_name VARCHAR NOT NULL,
    canonical_product_id VARCHAR NOT NULL,
    price_inr DECIMAL(18,2) NOT NULL,
    discount_pct INTEGER,
    is_focal BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_products (
    canonical_product_id VARCHAR PRIMARY KEY,
    brand VARCHAR,
    product_title VARCHAR,
    category_l1 VARCHAR,
    category_l2 VARCHAR,
    avg_price DECIMAL(18,2),
    first_seen TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_users (
    user_id VARCHAR PRIMARY KEY,
    first_seen TIMESTAMP,
    primary_app VARCHAR,
    total_events INTEGER
);


-- The product matcher's decisions, so a registry rebuilt from the warehouse resolves every
-- listing exactly as the load that wrote it did. alias_key is a listing's exact v1.0 key and
-- canonical_product_id the product it resolved to; the row with alias_key equal to
-- canonical_product_id is the product's registration. source_observation_id
-- (`<event_id>#<listing position>`) is the listing that first produced the key.
CREATE TABLE IF NOT EXISTS product_aliases (
    alias_key VARCHAR PRIMARY KEY,
    canonical_product_id VARCHAR NOT NULL,
    source_observation_id VARCHAR NOT NULL
);

-- One row per pair of consecutive views of the same product by the same user on two
-- different apps. primary_app is the app of the earlier event and compared_app the app of
-- the later event; first_viewed and last_viewed are their timestamps.
CREATE TABLE IF NOT EXISTS fact_cross_app_comparisons (
    comparison_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    session_id VARCHAR,
    canonical_product_id VARCHAR NOT NULL,
    primary_app VARCHAR NOT NULL,
    compared_app VARCHAR NOT NULL,
    first_viewed TIMESTAMP NOT NULL,
    last_viewed TIMESTAMP NOT NULL,
    duration_minutes DECIMAL(10,2) NOT NULL,
    price_first_inr DECIMAL(18,2),
    price_last_inr DECIMAL(18,2),
    cheaper_app VARCHAR
);


-- Indexes for accelerated analytical filtering and session joins
CREATE INDEX IF NOT EXISTS idx_fact_user_ts ON fact_events(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_product ON fact_events(canonical_product_id);
CREATE INDEX IF NOT EXISTS idx_fact_session ON fact_events(session_id);
CREATE INDEX IF NOT EXISTS idx_cross_app_user ON fact_cross_app_comparisons(user_id);
CREATE INDEX IF NOT EXISTS idx_price_obs_user_product ON fact_price_observations(user_id, canonical_product_id);
