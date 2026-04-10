-- Retail demand analysis queries

SELECT date, SUM(sales) AS total_sales
FROM retail_demand
GROUP BY date
ORDER BY date;

SELECT store_id, promo, AVG(sales) AS avg_sales
FROM retail_demand
GROUP BY store_id, promo
ORDER BY store_id, promo;

SELECT holiday, weekend, AVG(sales) AS avg_sales
FROM retail_demand
GROUP BY holiday, weekend
ORDER BY holiday, weekend;
