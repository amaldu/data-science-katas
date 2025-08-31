-- --- 9. Simple transactions table ---

-- Table
CREATE TABLE IF NOT EXISTS simple_transactions (
    user_id INT NOT NULL,
    spend FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO simple_transactions (user_id, spend, transaction_date)
VALUES
    (101, 49.99, NOW() - interval '20 days'),
    (102, 25.50, NOW() - interval '19 days'),
    (103, 15.75, NOW() - interval '18 days'),
    (104, 99.99, NOW() - interval '17 days'),
    (105, 45.00, NOW() - interval '16 days'),
    (106, 30.25, NOW() - interval '15 days'),
    (101, 12.50, NOW() - interval '14 days'),
    (102, 60.00, NOW() - interval '13 days'),
    (103, 20.00, NOW() - interval '12 days'),
    (104, 75.50, NOW() - interval '11 days'),
    (105, 40.00, NOW() - interval '10 days'),
    (106, 55.25, NOW() - interval '9 days'),
    (101, 33.33, NOW() - interval '8 days'),
    (102, 80.00, NOW() - interval '7 days'),
    (103, 22.50, NOW() - interval '6 days'),
    (104, 90.00, NOW() - interval '5 days'),
    (105, 44.44, NOW() - interval '4 days'),
    (106, 66.66, NOW() - interval '3 days'),
    (101, 77.77, NOW() - interval '2 days'),
    (102, 88.88, NOW() - interval '1 day');
