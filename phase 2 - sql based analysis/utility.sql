-- table created date

SELECT object_name AS table_name, created
FROM user_objects
WHERE object_type = 'TABLE'
  AND created >= TO_DATE('28-AUG-25','DD-MON-YY')
  AND created < TO_DATE('29-AUG-25','DD-MON-YY')
ORDER BY created DESC;