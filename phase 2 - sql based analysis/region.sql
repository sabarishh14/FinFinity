-- each customer region

CREATE OR REPLACE VIEW customer_regions AS
SELECT
    c.customer_id,
    c.cust_state,
    CASE 
        WHEN c.cust_state IN ('Connecticut','Delaware','Maine','Maryland','Massachusetts','Pennsylvania','Vermont')
             THEN 'Northeast'
        WHEN c.cust_state IN ('Illinois','Indiana','Iowa','Kansas','Michigan','Minnesota','Missouri','Nebraska','Ohio','Wisconsin')
             THEN 'Midwest'
        WHEN c.cust_state IN ('Alabama','Arkansas','Florida','Georgia','Kentucky','Louisiana','Mississippi','Oklahoma','Tennessee','Texas','Virginia')
             THEN 'South'
        WHEN c.cust_state IN ('Alaska','Arizona','California','Colorado','Hawaii','Idaho','Montana','Nevada','Oregon','Washington','Wyoming')
             THEN 'West'
        ELSE 'Other'
    END AS region
FROM
    customers c
ORDER BY c.customer_id;


select * from customer_regions;