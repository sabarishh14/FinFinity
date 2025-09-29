-- no_of_dependants - total_asset - avg_asset - customer_count
SELECT
    c.num_dependants,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    SUM(ca.total) AS total_assets,
    ROUND(AVG(ca.total), 2) AS avg_asset_value_per_assetobj,
    ROUND(SUM(ca.total) / NULLIF(COUNT(DISTINCT c.customer_id), 0), 2) AS avg_assets_per_customer
FROM
    customers c
LEFT JOIN customer_assets ca 
    ON c.customer_id = ca.customer_id
GROUP BY
    c.num_dependants
ORDER BY
    c.num_dependants;

-- no_of_dependants - highest_asset - lowest_asset - avg_asset - customer_count

WITH customer_total_assets AS (
    SELECT 
        c.customer_id,
        c.num_dependants,
        SUM(ca.total) AS total_assets
    FROM 
        customers c
    LEFT JOIN 
        customer_assets ca ON c.customer_id = ca.customer_id
    GROUP BY 
        c.customer_id, c.num_dependants
)
SELECT
    num_dependants,
    COUNT(customer_id) AS customer_count,
    MIN(total_assets) AS lowest_customer_asset,
    MAX(total_assets) AS highest_customer_asset,
    ROUND(AVG(total_assets), 2) AS avg_customer_asset
FROM
    customer_total_assets
GROUP BY
    num_dependants
ORDER BY
    num_dependants;


