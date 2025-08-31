-- --- 7. User transactions table ---

-- Table
CREATE TABLE IF NOT EXISTS user_transactions (
    transaction_id INT PRIMARY KEY,
    product_id INT NOT NULL,
    user_id INT NOT NULL,
    spend FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO user_transactions (transaction_id, product_id, user_id, spend, transaction_date)
VALUES
    (1, 101, 1, 49.99, NOW() - interval '8 days'),
    (2, 102, 2, 25.50, NOW() - interval '7 days'),
    (3, 103, 3, 15.75, NOW() - interval '6 days'),
    (4, 104, 4, 99.99, NOW() - interval '5 days'),
    (5, 101, 1, 45.00, NOW() - interval '4 days'),
    (6, 102, 2, 30.25, NOW() - interval '3 days'),
    (7, 103, 3, 12.50, NOW() - interval '2 days'),
    (8, 104, 4, 60.00, NOW() - interval '1 day'),
    (9, 101, 1, 20.00, NOW() - interval '8 days'),
    (10, 102, 2, 75.50, NOW() - interval '7 days'),
    (11, 103, 3, 40.00, NOW() - interval '6 days'),
    (12, 104, 4, 55.25, NOW() - interval '5 days'),
    (13, 101, 1, 33.33, NOW() - interval '4 days'),
    (14, 102, 2, 80.00, NOW() - interval '3 days'),
    (15, 103, 3, 22.50, NOW() - interval '2 days'),
    (16, 104, 4, 90.00, NOW() - interval '1 day'),
    (17, 101, 1, 44.44, NOW() - interval '8 days'),
    (18, 102, 2, 66.66, NOW() - interval '7 days'),
    (19, 103, 3, 77.77, NOW() - interval '6 days'),
    (20, 104, 4, 88.88, NOW());
