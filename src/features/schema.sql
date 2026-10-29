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
