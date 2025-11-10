WITH postal_codes AS (
    SELECT DISTINCT postal_code
    FROM levy_tax
    WHERE archived_at IS NULL
),
hs_codes AS (
    SELECT DISTINCT hs_code
    FROM levy
    WHERE date_deleted IS NULL
      AND country = 'US'
      AND hs_code IS NOT NULL AND hs_code <> ''
),
countries AS (
    SELECT DISTINCT country
    FROM levy
    WHERE date_deleted IS NULL
      AND country <> 'US'
      AND country <> 'XK'
),
-- Enumerate each option set
countries_enum AS (
    SELECT country, ROW_NUMBER() OVER (ORDER BY country) AS rn
    FROM countries
),
hs_enum AS (
    SELECT hs_code, ROW_NUMBER() OVER (ORDER BY hs_code) AS rn
    FROM hs_codes
),
postal_enum AS (
    SELECT postal_code, ROW_NUMBER() OVER (ORDER BY postal_code) AS rn
    FROM postal_codes
),
-- Sizes for modulo/rand ranges
sizes AS (
    SELECT
      (SELECT COUNT(*) FROM countries_enum) AS coo_sz,
      (SELECT COUNT(*) FROM hs_enum)        AS hs_sz,
      (SELECT COUNT(*) FROM postal_enum)    AS pc_sz
),
-- 10k row generator
digits AS (
    SELECT 0 AS d UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
    UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
),
nums AS (
    SELECT (d4.d * 1000 + d3.d * 100 + d2.d * 10 + d1.d) AS n
    FROM digits d1
    CROSS JOIN digits d2
    CROSS JOIN digits d3
    CROSS JOIN digits d4
),
-- Pick random indices ONCE per output row (avoid random in join predicates)
picks AS (
    SELECT
        n,
        /* 1..coo_sz, 1..hs_sz, 1..pc_sz */
        (FLOOR(RANDOM() * coo_sz)::int + 1)   AS coo_rn,
        (FLOOR(RANDOM() * coo_sz)::int + 1)   AS sfc_rn,
        (FLOOR(RANDOM() * hs_sz)::int + 1)    AS hs_rn,
        (FLOOR(RANDOM() * pc_sz)::int + 1)    AS pc_rn
    FROM nums
    CROSS JOIN sizes
    LIMIT 10000
)
SELECT
    coo.country  AS country_of_origin,
    sfc.country  AS ship_from_country,
    hs.hs_code,
    pc.postal_code AS ship_to_postal_code
FROM picks p
JOIN countries_enum coo ON coo.rn = p.coo_rn
JOIN countries_enum sfc ON sfc.rn = p.sfc_rn
JOIN hs_enum hs         ON hs.rn  = p.hs_rn
JOIN postal_enum pc     ON pc.rn  = p.pc_rn;
