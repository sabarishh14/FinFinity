-- each customer - no of qtns answered - percent split

CREATE OR REPLACE VIEW vw_customer_risk_distribution AS
WITH customer_risk_counts AS (
    SELECT
        ca.customer_id,
        a.risk_profile_id,
        COUNT(*) AS risk_count
    FROM
        customer_answers ca
        JOIN answers a ON ca.qa_id = a.qa_id
    WHERE ca.qa_id IS NOT NULL
    GROUP BY
        ca.customer_id, a.risk_profile_id
),
customer_totals AS (
    SELECT
        ca.customer_id,
        COUNT(*) AS answered_cnt
    FROM customer_answers ca
    WHERE ca.qa_id IS NOT NULL
    GROUP BY ca.customer_id
)
SELECT
    ct.customer_id,
    ct.answered_cnt,
 
    -- counts
    COALESCE(SUM(CASE WHEN crc.risk_profile_id = 1 THEN crc.risk_count END),0) AS low_ans_cnt,
    COALESCE(SUM(CASE WHEN crc.risk_profile_id = 2 THEN crc.risk_count END),0) AS below_ans_cnt,
    COALESCE(SUM(CASE WHEN crc.risk_profile_id = 3 THEN crc.risk_count END),0) AS avg_ans_cnt,
    COALESCE(SUM(CASE WHEN crc.risk_profile_id = 4 THEN crc.risk_count END),0) AS above_ans_cnt,
    COALESCE(SUM(CASE WHEN crc.risk_profile_id = 5 THEN crc.risk_count END),0) AS high_ans_cnt,
 
    -- percentages
    CASE WHEN ct.answered_cnt > 0
         THEN ROUND(COALESCE(SUM(CASE WHEN crc.risk_profile_id = 1 THEN crc.risk_count END),0) * 100.0 / ct.answered_cnt, 2)
         ELSE 0 END AS low_ans_per,
 
    CASE WHEN ct.answered_cnt > 0
         THEN ROUND(COALESCE(SUM(CASE WHEN crc.risk_profile_id = 2 THEN crc.risk_count END),0) * 100.0 / ct.answered_cnt, 2)
         ELSE 0 END AS below_ans_per,
 
    CASE WHEN ct.answered_cnt > 0
         THEN ROUND(COALESCE(SUM(CASE WHEN crc.risk_profile_id = 3 THEN crc.risk_count END),0) * 100.0 / ct.answered_cnt, 2)
         ELSE 0 END AS avg_ans_per,
 
    CASE WHEN ct.answered_cnt > 0
         THEN ROUND(COALESCE(SUM(CASE WHEN crc.risk_profile_id = 4 THEN crc.risk_count END),0) * 100.0 / ct.answered_cnt, 2)
         ELSE 0 END AS above_ans_per,
 
    CASE WHEN ct.answered_cnt > 0
         THEN ROUND(COALESCE(SUM(CASE WHEN crc.risk_profile_id = 5 THEN crc.risk_count END),0) * 100.0 / ct.answered_cnt, 2)
         ELSE 0 END AS high_ans_per
 
FROM
    customer_totals ct
    LEFT JOIN customer_risk_counts crc ON ct.customer_id = crc.customer_id
GROUP BY
    ct.customer_id, ct.answered_cnt
ORDER BY
    ct.customer_id;

select * from vw_customer_risk_distribution;

-- each customer - max answer - pct (all)

CREATE VIEW VW_CUSTOMER_MAX_RISK AS
SELECT
    customer_id,
    CASE 
        WHEN low_ans_per = GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) THEN 'LOW'
        WHEN below_ans_per = GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) THEN 'BELOW_AVG'
        WHEN avg_ans_per = GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) THEN 'AVG'
        WHEN above_ans_per = GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) THEN 'ABOVE_AVG'
        WHEN high_ans_per = GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) THEN 'HIGH'
    END AS max_risk_category,
    GREATEST(low_ans_per, below_ans_per, avg_ans_per, above_ans_per, high_ans_per) AS max_risk_percent
FROM
    vw_customer_risk_distribution;

select * from VW_CUSTOMER_MAX_RISK;

---- each customer max answer and max asset obj (40-60)

create or replace view vw_customer_max_ass_risk as
select c.customer_id,
ca.max_risk_category, c.MAX_ASSET_OBJECTIVE
from VW_CUSTOMER_MAX_ASS_PCT c
join  vw_customer_max_risk ca
on c.customer_id=ca.customer_id 
order by c.customer_id;

select * from VW_CUSTOMER_MAX_ASS_RISK;

--- final view -> asset obj and each answer pct (40-60) 

SELECT
    max_asset_objective AS asset_objective_name,
    COUNT(*) AS customer_count,
    ROUND(100 * SUM(CASE WHEN max_risk_category = 'LOW' THEN 1 ELSE 0 END) / COUNT(*), 2) AS low_risk_pct,
    ROUND(100 * SUM(CASE WHEN max_risk_category = 'BELOW_AVG' THEN 1 ELSE 0 END) / COUNT(*), 2) AS below_avg_pct,
    ROUND(100 * SUM(CASE WHEN max_risk_category = 'AVG' THEN 1 ELSE 0 END) / COUNT(*), 2) AS avg_pct,
    ROUND(100 * SUM(CASE WHEN max_risk_category = 'ABOVE_AVG' THEN 1 ELSE 0 END) / COUNT(*), 2) AS above_avg_pct,
    ROUND(100 * SUM(CASE WHEN max_risk_category = 'HIGH' THEN 1 ELSE 0 END) / COUNT(*), 2) AS high_risk_pct
FROM
    VW_CUSTOMER_MAX_ASS_RISK
GROUP BY
    max_asset_objective
ORDER BY
    customer_count DESC;

    -- marital_status - no_of_dependants - customer - answer_count

SELECT 
    c.marital_status,
    c.num_dependants,
    COUNT(DISTINCT c.customer_id) AS customer_count,  -- unique customers in this group
    COUNT(*) AS total_responses,

    -- Counts per risk profile
    SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) AS low_cnt,
    SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) AS below_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) AS avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) AS above_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) AS high_cnt,

    -- Percentages per risk profile (out of total responses for this group)
    ROUND(SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 2) AS low_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 2) AS below_avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 2) AS avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 2) AS above_avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*),0), 2) AS high_pct

FROM customers c
LEFT JOIN customer_answers ca ON c.customer_id = ca.customer_id
LEFT JOIN answers a ON ca.qa_id = a.qa_id
GROUP BY c.marital_status, c.num_dependants
ORDER BY c.marital_status, c.num_dependants;

-- marital_status - gender - customer - answer_count

SELECT 
    c.marital_status,
    c.gender,
    COUNT(DISTINCT c.customer_id) AS customer_count,  -- unique customers in this group
    COUNT(a.risk_profile_id) AS total_responses,      -- only count answered questions

    -- Counts per risk profile
    SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) AS low_cnt,
    SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) AS below_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) AS avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) AS above_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) AS high_cnt,

    -- Percentages per risk profile (out of total responses in that group)
    ROUND(SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2) AS low_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2) AS below_avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2) AS avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2) AS above_avg_pct,
    ROUND(SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2) AS high_pct

FROM customers c
LEFT JOIN customer_answers ca ON c.customer_id = ca.customer_id
LEFT JOIN answers a ON ca.qa_id = a.qa_id
GROUP BY c.marital_status, c.gender
ORDER BY c.marital_status, c.gender;


-- marital_status - gender - no_of_dependants - answer_count

SELECT 
    c.marital_status,
    c.gender,
    c.num_dependants,
    COUNT(DISTINCT c.customer_id) AS customer_count,   -- unique customers in this group
    COUNT(a.risk_profile_id) AS total_responses,       -- only answered questions

    -- Counts per risk profile
    SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) AS low_cnt,
    SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) AS below_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) AS avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) AS above_avg_cnt,
    SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) AS high_cnt,

    -- Percentages per risk profile (out of total responses in that group)
    COALESCE(ROUND(SUM(CASE WHEN a.risk_profile_id = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2), 0) AS low_pct,
    COALESCE(ROUND(SUM(CASE WHEN a.risk_profile_id = 2 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2), 0) AS below_avg_pct,
    COALESCE(ROUND(SUM(CASE WHEN a.risk_profile_id = 3 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2), 0) AS avg_pct,
    COALESCE(ROUND(SUM(CASE WHEN a.risk_profile_id = 4 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2), 0) AS above_avg_pct,
    COALESCE(ROUND(SUM(CASE WHEN a.risk_profile_id = 5 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(a.risk_profile_id), 0), 2), 0) AS high_pct

FROM customers c
LEFT JOIN customer_answers ca ON c.customer_id = ca.customer_id
LEFT JOIN answers a ON ca.qa_id = a.qa_id
GROUP BY c.marital_status, c.gender, c.num_dependants
ORDER BY c.marital_status, c.gender, c.num_dependants;