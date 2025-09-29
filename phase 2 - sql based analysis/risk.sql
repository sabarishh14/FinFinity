-- each customer - mean risk - median risk - mode risk

SELECT
    sub.customer_id,
    ROUND(AVG(sub.risk_profile_id), 0) AS mean_risk,
    MEDIAN(sub.risk_profile_id) AS median_risk,
    MAX(sub.risk_profile_id) KEEP (DENSE_RANK FIRST ORDER BY sub.cnt DESC) AS mode_risk
FROM (
    SELECT
        c.customer_id,
        a.risk_profile_id,
        COUNT(*) OVER (PARTITION BY c.customer_id, a.risk_profile_id) AS cnt
    FROM customers c
    JOIN customer_answers ca ON c.customer_id = ca.customer_id
    JOIN answers a ON ca.qa_id = a.qa_id
    -- WHERE c.age BETWEEN 40 AND 60
) sub
GROUP BY sub.customer_id;
-- HAVING COUNT(*) >= 5;

--- WEIGHTED AVERAGE

CREATE VIEW customer_risk_scores AS
WITH details AS (
    SELECT 
        c.customer_id,
        CASE
            WHEN c.age < 30 THEN '<30'
            WHEN c.age BETWEEN 30 AND 40 THEN '30-40'
            WHEN c.age BETWEEN 40 AND 50 THEN '40-50'
            WHEN c.age BETWEEN 50 AND 60 THEN '50-60'
            WHEN c.age BETWEEN 60 AND 70 THEN '60-70'
            ELSE '70+'
        END AS age_group,
        COUNT(a.qa_id) AS total_answers
    FROM 
        customers c
    JOIN 
        customer_answers ca ON c.customer_id = ca.customer_id
    JOIN 
        answers a ON ca.qa_id = a.qa_id
    GROUP BY 
        c.customer_id, 
        CASE
            WHEN c.age < 30 THEN '<30'
            WHEN c.age BETWEEN 30 AND 40 THEN '30-40'
            WHEN c.age BETWEEN 40 AND 50 THEN '40-50'
            WHEN c.age BETWEEN 50 AND 60 THEN '50-60'
            WHEN c.age BETWEEN 60 AND 70 THEN '60-70'
            ELSE '70+'
        END
    HAVING 
        COUNT(a.qa_id) >= 6
),
risk_data AS (
    SELECT 
        d.customer_id,
        d.age_group,
        a.risk_profile_id,
        COUNT(a.qa_id) AS risk_count
    FROM 
        details d
    JOIN 
        customer_answers ca ON d.customer_id = ca.customer_id
    JOIN 
        answers a ON ca.qa_id = a.qa_id
    GROUP BY 
        d.customer_id, d.age_group, a.risk_profile_id
),
weighted_averages AS (
    SELECT 
        r.customer_id,
        ROUND(SUM(r.risk_profile_id * r.risk_count * r.risk_count) / 
              SUM(r.risk_count * r.risk_count), 2) AS weighted_average
    FROM 
        risk_data r
    GROUP BY 
        r.customer_id
)
SELECT 
    customer_id,
    weighted_average AS risk_score
FROM 
    weighted_averages;
    
select * from customer_risk_scores;