-- --- 3. Total trans table --- 

-- Table
CREATE TABLE IF NOT EXISTS total_trans (
    user_id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id VARCHAR(25) NOT NULL, 
    spend NUMERIC(4, 2) NOT NULL, 
    trans_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO total_trans (user_id, order_id, product_id, spend, trans_date)
VALUES
(101, 1, 'product_A', 49.99, NOW() - interval '10 days'),
    (101, 7, 'product_B', 15.50, NOW() - interval '4 days'),
    (101, 13, 'product_F', 35.00, NOW() - interval '9 hours'),
    (101, 19, 'product_A', 30.00, NOW() - interval '3 hours'),
    (102, 2, 'product_B', 75.50, NOW() - interval '9 days'),
    (102, 8, 'product_C', 45.00, NOW() - interval '3 days'),
    (102, 14, 'product_A', 25.25, NOW() - interval '8 hours'),
    (102, 20, 'product_F', 45.50, NOW() - interval '2 hours'),
    (103, 3, 'product_C', 20.00, NOW() - interval '8 days'),
    (103, 9, 'product_A', 10.99, NOW() - interval '2 days'),
    (103, 15, 'product_B', 40.00, NOW() - interval '7 hours'),
    (104, 4, 'product_D', 100.00, NOW() - interval '7 days'),
    (104, 10, 'product_E', 70.75, NOW() - interval '1 day'),
    (104, 16, 'product_C', 55.50, NOW() - interval '6 hours'),
    (105, 5, 'product_E', 60.25, NOW() - interval '6 days'),
    (105, 11, 'product_D', 50.00, NOW() - interval '12 hours'),
    (105, 17, 'product_E', 65.00, NOW() - interval '5 hours'),
    (106, 6, 'product_F', 33.33, NOW() - interval '5 days'),
    (106, 12, 'product_C', 22.50, NOW() - interval '10 hours'),
    (106, 18, 'product_D', 80.00, NOW() - interval '4 hours');

