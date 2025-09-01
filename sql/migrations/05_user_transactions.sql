-- --- 5. User transactions table --- 
DROP TABLE IF EXISTS user_transactions;

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS user_transactions (
    transaction_id INT NOT NULL,
    user_id UUID PRIMARY KEY DEFAULT geN_random_uuid(),
    product_id VARCHAR(25) NOT NULL, 
    spend NUMERIC(6, 2) NOT NULL, 
    trans_date TIMESTAMP NOT NULL
);


-- Values
INSERT INTO user_transactions (transaction_id, product_id, spend, trans_date)
VALUES
    (5, 'product_B', 75.50, NOW() - interval '49 days'),
    (12, 'product_L', 15.50, NOW() - interval '39 days'),
    (3, 'product_C', 20.00, NOW() - interval '48 days'),
    (18, 'product_R', 30.00, NOW() - interval '33 days'),
    (21, 'product_U', 44.50, NOW() - interval '30 days'),
    (8, 'product_H', 55.50, NOW() - interval '43 days'),
    (14, 'product_N', 85.25, NOW() - interval '37 days'),
    (7, 'product_G', 80.00, NOW() - interval '44 days'),
    (2, 'product_B', 25.50, NOW() - interval '19 days'),
    (20, 'product_T', 12.99, NOW() - interval '31 days'),
    (13, 'product_M', 70.00, NOW() - interval '38 days'),
    (1, 'product_A', 49.99, NOW() - interval '50 days'),
    (6, 'product_F', 33.33, NOW() - interval '45 days'),
    (17, 'product_Q', 50.50, NOW() - interval '34 days'),
    (24, 'product_X', 19.99, NOW() - interval '27 days'),
    (19, 'product_S', 95.75, NOW() - interval '32 days'),
    (10, 'product_J', 90.10, NOW() - interval '41 days'),
    (23, 'product_W', 88.00, NOW() - interval '28 days'),
    (16, 'product_P', 65.00, NOW() - interval '35 days'),
    (22, 'product_V', 27.75, NOW() - interval '29 days'),
    (4, 'product_D', 100.00, NOW() - interval '47 days'),
    (11, 'product_K', 25.00, NOW() - interval '40 days'),
    (25, 'product_Y', 55.25, NOW() - interval '26 days'),
    (9, 'product_I', 42.75, NOW() - interval '42 days'),
    (15, 'product_O', 40.00, NOW() - interval '36 days'),
    (26, 'product_Z', 71.00, NOW() - interval '25 days'),
    (28, 'product_B', 62.75, NOW() - interval '23 days'),
    (27, 'product_A', 38.50, NOW() - interval '24 days'),
    (29, 'product_C', 45.00, NOW() - interval '22 days'),
    (30, 'product_D', 90.00, NOW() - interval '21 days'),
    (31, 'product_E', 33.25, NOW() - interval '20 days'),
    (32, 'product_F', 49.50, NOW() - interval '19 days'),
    (33, 'product_G', 58.00, NOW() - interval '18 days'),
    (34, 'product_H', 77.25, NOW() - interval '17 days'),
    (35, 'product_I', 66.50, NOW() - interval '16 days'),
    (36, 'product_J', 80.00, NOW() - interval '15 days'),
    (37, 'product_K', 22.75, NOW() - interval '14 days'),
    (38, 'product_L', 18.50, NOW() - interval '13 days'),
    (39, 'product_M', 47.00, NOW() - interval '12 days'),
    (40, 'product_N', 53.25, NOW() - interval '11 days'),
    (41, 'product_O', 35.00, NOW() - interval '10 days'),
    (42, 'product_P', 41.50, NOW() - interval '9 days'),
    (43, 'product_Q', 68.75, NOW() - interval '8 days'),
    (44, 'product_R', 59.99, NOW() - interval '7 days'),
    (45, 'product_S', 49.50, NOW() - interval '6 days'),
    (46, 'product_T', 72.25, NOW() - interval '5 days'),
    (47, 'product_U', 33.75, NOW() - interval '4 days'),
    (48, 'product_V', 46.50, NOW() - interval '3 days'),
    (49, 'product_W', 80.00, NOW() - interval '2 days'),
    (50, 'product_X', 55.25, NOW() - interval '1 day');
