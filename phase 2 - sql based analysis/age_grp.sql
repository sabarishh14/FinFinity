-- age_grp - risk_profile - answers pct

CREATE or replace VIEW AGE_GRP_RISK_CAT AS
SELECT age_group, risk_profile, COUNT(*) AS cnt, 
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY age_group), 2) AS pct
FROM (
    SELECT
        CASE
            WHEN c.age < 30 THEN '<30'
            WHEN c.age BETWEEN 30 AND 39 THEN '30-39'
            WHEN c.age BETWEEN 40 AND 49 THEN '40-49'
            WHEN c.age BETWEEN 50 AND 59 THEN '50-59'
            WHEN c.age BETWEEN 60 AND 69 THEN '60-69'
            ELSE '70+'
        END AS age_group,
        r.risk_profile
    FROM CUSTOMERS c
    JOIN CUSTOMER_ANSWERS ca ON c.CUSTOMER_ID = ca.CUSTOMER_ID
    JOIN ANSWERS a ON ca.QA_ID = a.QA_ID
    JOIN RISK_PROFILE r ON a.RISK_PROFILE_ID = r.RISK_PROFILE_ID
) t
GROUP BY age_group, risk_profile;

select * from AGE_GRP_RISK_CAT;