-- customers from 40-60 - percent split for each asset objective
CREATE OR REPLACE VIEW vw_customer_asset_pct_40_60 AS
SELECT 
    c.customer_id,
    c.age,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Family Planning' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS family_planning_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'General Savings' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS general_savings_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Home Ownership' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS home_ownership_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Retirement' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS retirement_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Unknown' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS unknown_pct
FROM 
    customers c
LEFT JOIN customer_assets ca 
    ON c.customer_id = ca.customer_id
LEFT JOIN asset_objectives ao 
    ON ca.asset_objective_id = ao.asset_objective_id
WHERE 
    c.age BETWEEN 40 AND 60
GROUP BY 
    c.customer_id, c.age;

SELECT * FROM VW_CUSTOMER_ASSET_PCT_40_60;

-- customers from 60-80 - percent split for each asset objective
CREATE OR REPLACE VIEW vw_customer_asset_pct_60_80 AS
SELECT 
    c.customer_id,
    c.age,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Family Planning' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS family_planning_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'General Savings' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS general_savings_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Home Ownership' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS home_ownership_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Retirement' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS retirement_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Unknown' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS unknown_pct
FROM 
    customers c
LEFT JOIN customer_assets ca 
    ON c.customer_id = ca.customer_id
LEFT JOIN asset_objectives ao 
    ON ca.asset_objective_id = ao.asset_objective_id
WHERE 
    c.age BETWEEN 60 AND 80
GROUP BY 
    c.customer_id, c.age;

select * from VW_CUSTOMER_ASSET_PCT_60_80;

-- customers from 60-80 - max percent in which asset objective
CREATE or replace VIEW VW_CUSTOMER_MAX_ASS_PCT AS SELECT 
    v.customer_id,
    v.age,
    CASE 
        WHEN v.family_planning_pct = GREATEST(
                NVL(v.family_planning_pct,0),
                NVL(v.general_savings_pct,0),
                NVL(v.home_ownership_pct,0),
                NVL(v.retirement_pct,0),
                NVL(v.unknown_pct,0)) THEN 'Family Planning'
        WHEN v.general_savings_pct = GREATEST(
                NVL(v.family_planning_pct,0),
                NVL(v.general_savings_pct,0),
                NVL(v.home_ownership_pct,0),
                NVL(v.retirement_pct,0),
                NVL(v.unknown_pct,0)) THEN 'General Savings'
        WHEN v.home_ownership_pct = GREATEST(
                NVL(v.family_planning_pct,0),
                NVL(v.general_savings_pct,0),
                NVL(v.home_ownership_pct,0),
                NVL(v.retirement_pct,0),
                NVL(v.unknown_pct,0)) THEN 'Home Ownership'
        WHEN v.retirement_pct = GREATEST(
                NVL(v.family_planning_pct,0),
                NVL(v.general_savings_pct,0),
                NVL(v.home_ownership_pct,0),
                NVL(v.retirement_pct,0),
                NVL(v.unknown_pct,0)) THEN 'Retirement'
        WHEN v.unknown_pct = GREATEST(
                NVL(v.family_planning_pct,0),
                NVL(v.general_savings_pct,0),
                NVL(v.home_ownership_pct,0),
                NVL(v.retirement_pct,0),
                NVL(v.unknown_pct,0)) THEN 'Unknown'
    END AS max_asset_objective,
    GREATEST(
        NVL(v.family_planning_pct,0),
        NVL(v.general_savings_pct,0),
        NVL(v.home_ownership_pct,0),
        NVL(v.retirement_pct,0),
        NVL(v.unknown_pct,0)) AS max_pct
FROM 
    vw_customer_asset_pct_60_80 v;

select * from VW_CUSTOMER_MAX_ASS_PCT;

-- all customers - percent split for each asset objective

CREATE OR REPLACE VIEW vw_customer_asset_pct AS
SELECT 
    c.customer_id,
    c.age,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Family Planning' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS family_planning_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'General Savings' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS general_savings_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Home Ownership' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS home_ownership_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Retirement' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS retirement_pct,
    ROUND(SUM(CASE WHEN ao.asset_objective_name = 'Unknown' 
                   THEN ca.total ELSE 0 END) * 100 / NULLIF(SUM(ca.total),0), 2) AS unknown_pct
FROM 
    customers c
LEFT JOIN customer_assets ca 
    ON c.customer_id = ca.customer_id
LEFT JOIN asset_objectives ao 
    ON ca.asset_objective_id = ao.asset_objective_id
GROUP BY 
    c.customer_id, c.age;
    
select * from VW_CUSTOMER_ASSET_PCT;

-- customers from 60-80 - customer_count -asset_objective - percent_count

SELECT
    v.max_asset_objective,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percent_of_total,
    ac.risk_profile_id
FROM
    VW_CUSTOMER_MAX_ASS_PCT v
LEFT JOIN asset_classes ac
    ON ac.asset_class = v.max_asset_objective
GROUP BY
    v.max_asset_objective,
    ac.risk_profile_id
ORDER BY
    customer_count DESC;