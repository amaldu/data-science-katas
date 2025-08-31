-- --- 5. User transactions table --- 

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS user_transactions (
    transaction_id INT PRIMARY KEY,
    user_id UUID NOT NULL DEFAULT get_random_uuid(),
    product_id VARCHAR(25) NOT NULL, 
    spend NUMERIC(6, 2) NOT NULL, 
    trans_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO user_transactions (user_id, product_id, spend, trans_date)
VALUES
    (1, gen_random_uuid(), 'product_A', 49.99, NOW() - interval '15 days'),
    (2, gen_random_uuid(), 'product_B', 75.50, NOW() - interval '14 days'),
    (3, gen_random_uuid(), 'product_C', 20.00, NOW() - interval '13 days'),
    (4, gen_random_uuid(), 'product_D', 100.00, NOW() - interval '12 days'),
    (5, gen_random_uuid(), 'product_E', 60.25, NOW() - interval '11 days'),
    (6, gen_random_uuid(), 'product_F', 33.33, NOW() - interval '10 days'),
    (7, gen_random_uuid(), 'product_G', 80.00, NOW() - interval '9 days'),
    (8, gen_random_uuid(), 'product_H', 55.50, NOW() - interval '8 days'),
    (9, gen_random_uuid(), 'product_I', 42.75, NOW() - interval '7 days'),
    (10, gen_random_uuid(), 'product_J', 90.10, NOW() - interval '6 days'),
    (11, gen_random_uuid(), 'product_K', 25.00, NOW() - interval '5 days'),
    (12, gen_random_uuid(), 'product_L', 15.50, NOW() - interval '4 days'),
    (13, gen_random_uuid(), 'product_M', 70.00, NOW() - interval '3 days'),
    (14, gen_random_uuid(), 'product_N', 85.25, NOW() - interval '2 days'),
    (15, gen_random_uuid(), 'product_O', 40.00, NOW() - interval '1 day');