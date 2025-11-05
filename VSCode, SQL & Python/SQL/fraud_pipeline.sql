-- ===============================================================
-- 📘 SMART BUDGET / Risk & Fraud Detection — Data Preparation SQL
-- Author: ChatGPT (adapted for Aïmane)
-- Date: 2025-11-03
-- ---------------------------------------------------------------
-- This script prepares clean, analytics-ready data for Power BI
-- and machine learning. It takes raw transaction data from a
-- credit card dataset and:
--   ✅ Cleans duplicates
--   ✅ Handles missing values
--   ✅ Creates time-based features
--   ✅ Simulates card IDs
--   ✅ Builds a final "transactions_clean" table
-- ===============================================================


-- === 0. PREPARATION ==================================================

-- Create a separate "fraud" schema to keep all our tables organized
CREATE SCHEMA IF NOT EXISTS fraud;

-- Make sure all following commands automatically use the "fraud" schema
SET search_path = fraud, public;

-- Remove old versions of the same tables (if we re-run this script)
DROP TABLE IF EXISTS fraud.transactions_clean CASCADE;
DROP TABLE IF EXISTS fraud.transactions_raw CASCADE;
DROP TABLE IF EXISTS fraud._staging_step1;
DROP TABLE IF EXISTS fraud._staging_step2;
DROP TABLE IF EXISTS fraud._staging_step3;
DROP TABLE IF EXISTS fraud._tmp_dedup;
DROP TABLE IF EXISTS fraud._medians;


-- === 1. CREATE THE RAW DATA TABLE ====================================

-- The dataset (from Kaggle) has 31 columns:
--   "Time", "V1"..."V28", "Amount", and "Class"
-- We'll store them in a table called transactions_raw
CREATE TABLE fraud.transactions_raw (
    id SERIAL PRIMARY KEY,
    time_seconds DOUBLE PRECISION,  -- The time since the first transaction
    v1 DOUBLE PRECISION, v2 DOUBLE PRECISION, v3 DOUBLE PRECISION, v4 DOUBLE PRECISION,
    v5 DOUBLE PRECISION, v6 DOUBLE PRECISION, v7 DOUBLE PRECISION, v8 DOUBLE PRECISION,
    v9 DOUBLE PRECISION, v10 DOUBLE PRECISION, v11 DOUBLE PRECISION, v12 DOUBLE PRECISION,
    v13 DOUBLE PRECISION, v14 DOUBLE PRECISION, v15 DOUBLE PRECISION, v16 DOUBLE PRECISION,
    v17 DOUBLE PRECISION, v18 DOUBLE PRECISION, v19 DOUBLE PRECISION, v20 DOUBLE PRECISION,
    v21 DOUBLE PRECISION, v22 DOUBLE PRECISION, v23 DOUBLE PRECISION, v24 DOUBLE PRECISION,
    v25 DOUBLE PRECISION, v26 DOUBLE PRECISION, v27 DOUBLE PRECISION, v28 DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    class INTEGER,  -- 0 = non-fraud, 1 = fraud
    raw_loaded_at TIMESTAMP WITH TIME ZONE DEFAULT now()  -- timestamp when data was loaded
);

-- === 2. LOAD THE RAW CSV =============================================

-- Load the data file from your computer.
-- ⚠️ IMPORTANT: Update the file path below to match your environment.
COPY fraud.transactions_raw(time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount, class)
FROM 'C:\Users\ayman\OneDrive\Desktop\New folder (3)\VSCode, SQL & Python\CSV\creditcard.csv'
WITH (FORMAT csv, HEADER true);


-- === 3. REMOVE DUPLICATES ============================================

-- Sometimes, the same transaction might appear multiple times.
-- This step removes exact duplicates while keeping one copy.

CREATE TABLE fraud._tmp_dedup AS
SELECT DISTINCT ON (time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount, class)
        *
FROM fraud.transactions_raw
ORDER BY time_seconds, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount, class, id;

-- Replace the old raw table with this clean version
DROP TABLE fraud.transactions_raw;
ALTER TABLE fraud._tmp_dedup RENAME TO transactions_raw;


-- === 4. HANDLE MISSING VALUES ========================================

-- Even though this dataset usually has no missing values,
-- this section shows how to safely fill any nulls using medians.

-- Calculate median (middle) values for each numeric column
CREATE TEMP TABLE _medians AS
SELECT
  percentile_disc(0.5) WITHIN GROUP (ORDER BY v1) AS v1,
  percentile_disc(0.5) WITHIN GROUP (ORDER BY v2) AS v2,
  ...
  percentile_disc(0.5) WITHIN GROUP (ORDER BY amount) AS amount_median
FROM fraud.transactions_raw;

-- Replace missing values with the median values calculated above
CREATE TABLE fraud._staging_step1 AS
SELECT
  id,
  time_seconds,
  COALESCE(v1, (SELECT v1 FROM _medians)) AS v1,
  ...
  COALESCE(amount, (SELECT amount_median FROM _medians)) AS amount,
  COALESCE(class, 0)::INTEGER AS class,  -- If class missing, assume it's non-fraud (0)
  raw_loaded_at
FROM fraud.transactions_raw;


-- === 5. ADD REALISTIC TIME COLUMN ====================================

-- The dataset’s “Time” column is just seconds since the first transaction.
-- Let’s convert that to a real timestamp to make trend analysis possible.

CREATE TABLE fraud._staging_step2 AS
SELECT
  id,
  time_seconds,
  timestamp with time zone '2013-01-01 00:00:00+00' + (time_seconds || ' seconds')::interval AS transaction_ts,
  *,
  class,
  raw_loaded_at
FROM fraud._staging_step1;


-- === 6. CREATE SYNTHETIC CARD IDs ====================================

-- The dataset is anonymous (no card numbers).
-- We simulate unique card IDs (card_0001, card_0002, …)
-- based on transaction patterns to represent individual users.

CREATE TABLE fraud._staging_step3 AS
SELECT
  *,
  ((row_number() OVER (ORDER BY md5(CAST(time_seconds as text) || '|' || CAST(amount as text))) - 1) % 2000)::integer AS card_bucket
FROM fraud._staging_step2
ORDER BY id;

-- Add readable card_id text column (e.g., “card_0456”)
ALTER TABLE fraud._staging_step3 ADD COLUMN card_id TEXT;
UPDATE fraud._staging_step3
SET card_id = 'card_' || LPAD(card_bucket::text, 4, '0');


-- === 7. FEATURE ENGINEERING ==========================================

-- Here we create new columns that help detect fraud:
--   🕒 transaction_hour — hour of the transaction
--   🌙 is_night_transaction — TRUE if between midnight and 5 AM
--   💰 transaction_amount_log — log-transformed amount to handle skewed data
--   📊 transactions_per_user_last_24h — how active the card was recently
--   💵 avg_amount_last_7days — typical spend for this card (for anomaly detection)

CREATE TABLE fraud.transactions_clean AS
WITH base AS (
  SELECT
    id,
    transaction_ts,
    card_id,
    card_bucket,
    v1,v2,...,v28,
    amount,
    class,
    EXTRACT(hour FROM transaction_ts)::int AS transaction_hour,
    CASE WHEN EXTRACT(hour FROM transaction_ts)::int BETWEEN 0 AND 5 THEN TRUE ELSE FALSE END AS is_night_transaction,
    ln(amount + 1) AS transaction_amount_log
  FROM fraud._staging_step3
)
SELECT
    b.*,
    COUNT(*) OVER (
      PARTITION BY card_id
      ORDER BY transaction_ts
      RANGE BETWEEN '24 hours' PRECEDING AND CURRENT ROW
    ) AS transactions_per_user_last_24h,
    COALESCE(
      AVG(amount) OVER (
        PARTITION BY card_id
        ORDER BY transaction_ts
        RANGE BETWEEN '7 days' PRECEDING AND '1 second' PRECEDING
      ),
    0.0) AS avg_amount_last_7days
FROM base b;


-- Add indexes (to make queries and dashboards faster)
CREATE INDEX ON fraud.transactions_clean (transaction_ts);
CREATE INDEX ON fraud.transactions_clean (card_id);
CREATE INDEX ON fraud.transactions_clean (class);
CREATE INDEX ON fraud.transactions_clean (transaction_hour);


-- === 8. CLEAN UP TEMPORARY TABLES ====================================

-- These intermediate tables were only needed for processing.
-- Let’s remove them to keep the database clean and light.
DROP TABLE IF EXISTS fraud._staging_step1, fraud._staging_step2, fraud._staging_step3, fraud._medians;

-- ✅ FINAL RESULT:
-- The clean table "fraud.transactions_clean" is ready for Power BI
-- It includes realistic timestamps, synthetic users, and engineered features.
-- You can now export it to CSV or connect Power BI directly to PostgreSQL.
