-- Star Schema DDL for Consumer Telemetry Data Warehouse

CREATE TABLE IF NOT EXISTS fact_events (
    event_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    app_name VARCHAR NOT NULL,
    canonical_product_id VARCHAR,
    action VARCHAR NOT NULL,
    price DECIMAL(10,2),
    currency VARCHAR,
    session_id VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_products (
    canonical_product_id VARCHAR PRIMARY KEY,
    brand VARCHAR,
    product_title VARCHAR,
    category_l1 VARCHAR,
    category_l2 VARCHAR,
    avg_price DECIMAL(10,2)
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
