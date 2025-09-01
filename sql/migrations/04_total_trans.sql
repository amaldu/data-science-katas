-- --- 3. Total trans table --- 
DROP TABLE IF EXISTS total_trans;

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS total_trans (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id INT NOT NULL,
    product_id VARCHAR(25) NOT NULL, 
    spend NUMERIC(6, 2) NOT NULL, 
    trans_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO total_trans (order_id, product_id, spend, trans_date) VALUES
(1, 'product_A', 49.99, NOW() - interval '50 days'),
(2, 'product_B', 75.50, NOW() - interval '49 days'),
(3, 'product_C', 20.00, NOW() - interval '48 days'),
(4, 'product_D', 100.00, NOW() - interval '47 days'),
(5, 'product_E', 60.25, NOW() - interval '46 days'),
(6, 'product_F', 33.33, NOW() - interval '45 days'),
(7, 'product_B', 15.50, NOW() - interval '44 days'),
(8, 'product_C', 45.00, NOW() - interval '43 days'),
(9, 'product_A', 10.99, NOW() - interval '42 days'),
(10, 'product_E', 70.75, NOW() - interval '41 days'),
(11, 'product_D', 50.00, NOW() - interval '40 days'),
(12, 'product_C', 22.50, NOW() - interval '39 days'),
(13, 'product_F', 35.00, NOW() - interval '38 days'),
(14, 'product_A', 25.25, NOW() - interval '37 days'),
(15, 'product_B', 40.00, NOW() - interval '36 days'),
(16, 'product_C', 55.50, NOW() - interval '35 days'),
(17, 'product_E', 65.00, NOW() - interval '34 days'),
(18, 'product_D', 80.00, NOW() - interval '33 days'),
(19, 'product_A', 30.00, NOW() - interval '32 days'),
(20, 'product_F', 45.50, NOW() - interval '31 days'),
(21, 'product_B', 38.75, NOW() - interval '30 days'),
(22, 'product_C', 27.50, NOW() - interval '29 days'),
(23, 'product_D', 90.00, NOW() - interval '28 days'),
(24, 'product_E', 60.00, NOW() - interval '27 days'),
(25, 'product_F', 35.50, NOW() - interval '26 days'),
(26, 'product_A', 42.00, NOW() - interval '25 days'),
(27, 'product_B', 48.00, NOW() - interval '24 days'),
(28, 'product_C', 53.25, NOW() - interval '23 days'),
(29, 'product_D', 66.00, NOW() - interval '22 days'),
(30, 'product_E', 75.50, NOW() - interval '21 days'),
(31, 'product_F', 32.75, NOW() - interval '20 days'),
(32, 'product_A', 29.99, NOW() - interval '19 days'),
(33, 'product_B', 41.00, NOW() - interval '18 days'),
(34, 'product_C', 36.50, NOW() - interval '17 days'),
(35, 'product_D', 79.99, NOW() - interval '16 days'),
(36, 'product_E', 68.50, NOW() - interval '15 days'),
(37, 'product_F', 33.33, NOW() - interval '14 days'),
(38, 'product_A', 47.00, NOW() - interval '13 days'),
(39, 'product_B', 55.50, NOW() - interval '12 days'),
(40, 'product_C', 61.25, NOW() - interval '11 days'),
(41, 'product_D', 49.99, NOW() - interval '10 days'),
(42, 'product_E', 52.50, NOW() - interval '9 days'),
(43, 'product_F', 38.75, NOW() - interval '8 days'),
(45, 'product_A', 41.25, NOW() - interval '7 days'),
(46, 'product_B', 36.50, NOW() - interval '6 days'),
(47, 'product_C', 55.75, NOW() - interval '5 days'),
(48, 'product_D', 62.00, NOW() - interval '4 days'),
(49, 'product_E', 47.50, NOW() - interval '3 days'),
(50, 'product_F', 39.99, NOW() - interval '2 days'),
(44, 'product_F', 20.99, NOW() - interval '2 days');
