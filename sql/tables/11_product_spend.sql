-- --- 10. Product spend table ---

-- Table
CREATE TABLE IF NOT EXISTS product_spend (
    transaction_id INT PRIMARY KEY,
    category_id INT NOT NULL,
    product_id INT NOT NULL,
    user_id INT NOT NULL,
    spend FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO product_spend (transaction_id, category_id, product_id, user_id, spend, transaction_date)
VALUES
    (1, 10, 101, 201, 49.99, NOW() - interval '20 days'),
    (2, 10, 102, 202, 25.50, NOW() - interval '19 days'),
    (3, 11, 103, 203, 15.75, NOW() - interval '18 days'),
    (4, 11, 104, 204, 99.99, NOW() - interval '17 days'),
    (5, 12, 105, 205, 45.00, NOW() - interval '16 days'),
    (6, 12, 106, 206, 30.25, NOW() - interval '15 days'),
    (7, 13, 101, 201, 12.50, NOW() - interval '14 days'),
    (8, 13, 102, 202, 60.00, NOW() - interval '13 days'),
    (9, 14, 103, 203, 20.00, NOW() - interval '12 days'),
    (10, 14, 104, 204, 75.50, NOW() - interval '11 days'),
    (11, 15, 105, 205, 40.00, NOW() - interval '10 days'),
    (12, 15, 106, 206, 55.25, NOW() - interval '9 days'),
    (13, 16, 101, 201, 33.33, NOW() - interval '8 days'),
    (14, 16, 102, 202, 80.00, NOW() - interval '7 days'),
    (15, 17, 103, 203, 22.50, NOW() - interval '6 days'),
    (16, 17, 104, 204, 90.00, NOW() - interval '5 days'),
    (17, 18, 105, 205, 44.44, NOW() - interval '4 days'),
    (18, 18, 106, 206, 66.66, NOW() - interval '3 days'),
    (19, 19, 101, 201, 77.77, NOW() - interval '2 days'),
    (20, 19, 102, 202, 88.88, NOW() - interval '1 day');
