-- age_group - risk_group - customer_count - avg_asset - qtns >=5

WITH answer_counts AS (
    SELECT 
        ca.customer_id, 
        COUNT(ca.qa_id) AS total_answers
    FROM 
        customer_answers ca
    GROUP BY 
        ca.customer_id
),
asset_totals AS (
    SELECT 
        ca.customer_id, 
        SUM(ca.total) AS total_asset_value
    FROM 
        customer_assets ca
    GROUP BY 
        ca.customer_id
),
details AS (
    SELECT 
        c.customer_id,
        at.total_asset_value,
        CASE
            WHEN c.age >= 30 AND c.age < 55 THEN '30-55'
            WHEN c.age >= 55 THEN '55+'
            ELSE NULL
        END AS age_group,
        ac.total_answers
    FROM 
        customers c
    JOIN 
        answer_counts ac ON c.customer_id = ac.customer_id
    JOIN 
        asset_totals at ON c.customer_id = at.customer_id
    WHERE 
        ac.total_answers >= 5
),
risk_data AS (
    SELECT 
        d.customer_id,
        d.age_group,
        d.total_asset_value,
        a.risk_profile_id,
        COUNT(*) AS risk_count
    FROM 
        details d
    JOIN 
        customer_answers ca ON d.customer_id = ca.customer_id
    JOIN 
        answers a ON ca.qa_id = a.qa_id
    GROUP BY 
        d.customer_id, d.age_group, d.total_asset_value, a.risk_profile_id
),
weighted_averages AS (
    SELECT 
        r.customer_id,
        r.age_group,
        r.total_asset_value,
        ROUND(SUM(r.risk_profile_id * POWER(r.risk_count, 2)) / SUM(POWER(r.risk_count, 2)), 2) AS weighted_average
    FROM 
        risk_data r
    GROUP BY 
        r.customer_id, r.age_group, r.total_asset_value
),
bucketed_data AS (
    SELECT 
        age_group,
        weighted_average,
        total_asset_value,
        CASE
            WHEN weighted_average >= 1.12 AND weighted_average < 2.34 THEN '1.12 - 2.34'
            WHEN weighted_average >= 2.34 AND weighted_average < 2.95 THEN '2.34 - 2.95'
            WHEN weighted_average >= 2.95 AND weighted_average < 3.56 THEN '2.95 - 3.56'
            WHEN weighted_average >= 3.56 AND weighted_average < 4.17 THEN '3.56 - 4.17'
            WHEN weighted_average >= 4.17 AND weighted_average <= 5 THEN '4.17 - 5'
            ELSE 'Out of Range'
        END AS risk_bucket
    FROM 
        weighted_averages
)
SELECT 
    age_group,
    risk_bucket,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_asset_value), 2) AS avg_asset_value
FROM 
    bucketed_data
WHERE 
    risk_bucket != 'Out of Range'
    AND age_group IN ('30-55', '55+')
GROUP BY 
    age_group, risk_bucket
ORDER BY 
    age_group,
    risk_bucket;