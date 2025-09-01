-- --- 10. Simple transactions table ---
DROP TABLE IF EXISTS simple_transactions;

-- Table
CREATE TABLE IF NOT EXISTS simple_transactions (
    user_id INT PRIMARY KEY NOT NULL,
    spend FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO simple_transactions (user_id, spend, transaction_date)
VALUES
    (107, 88.50, NOW() - interval '3 days'),
    (115, 45.20, NOW() - interval '12 days'),
    (102, 77.75, NOW() - interval '20 days'),
    (130, 66.30, NOW() - interval '7 days'),
    (121, 35.00, NOW() - interval '18 days'),
    (104, 92.10, NOW() - interval '25 days'),
    (119, 50.75, NOW() - interval '5 days'),
    (110, 28.40, NOW() - interval '10 days'),
    (125, 60.50, NOW() - interval '14 days'),
    (116, 40.00, NOW() - interval '1 day'),
    (101, 55.25, NOW() - interval '22 days'),
    (117, 30.10, NOW() - interval '6 days'),
    (108, 48.90, NOW() - interval '17 days'),
    (123, 75.00, NOW() - interval '8 days'),
    (105, 20.50, NOW() - interval '15 days'),
    (112, 85.40, NOW() - interval '2 days'),
    (131, 33.75, NOW() - interval '21 days'),
    (118, 44.60, NOW() - interval '13 days'),
    (113, 90.00, NOW() - interval '4 days'),
    (106, 25.30, NOW() - interval '19 days'),
    (120, 60.75, NOW() - interval '11 days'),
    (124, 37.50, NOW() - interval '16 days'),
    (103, 50.00, NOW() - interval '9 days'),
    (111, 28.90, NOW() - interval '23 days'),
    (109, 65.40, NOW() - interval '24 days'),
    (122, 55.10, NOW() - interval '26 days'),
    (126, 48.25, NOW() - interval '27 days'),
    (127, 33.00, NOW() - interval '28 days'),
    (128, 75.50, NOW() - interval '29 days'),
    (129, 80.25, NOW() - interval '30 days'),
    (132, 42.00, NOW() - interval '31 days'),
    (133, 36.50, NOW() - interval '32 days'),
    (134, 58.90, NOW() - interval '33 days'),
    (135, 29.75, NOW() - interval '34 days'),
    (136, 62.20, NOW() - interval '35 days'),
    (137, 45.80, NOW() - interval '36 days'),
    (138, 70.00, NOW() - interval '37 days'),
    (139, 55.50, NOW() - interval '38 days'),
    (140, 47.25, NOW() - interval '39 days'),
    (141, 63.10, NOW() - interval '40 days'),
    (142, 50.75, NOW() - interval '41 days'),
    (143, 38.40, NOW() - interval '42 days'),
    (144, 68.90, NOW() - interval '43 days'),
    (145, 53.25, NOW() - interval '44 days'),
    (146, 46.60, NOW() - interval '45 days'),
    (147, 59.00, NOW() - interval '46 days'),
    (148, 61.50, NOW() - interval '47 days'),
    (149, 72.30, NOW() - interval '48 days'),
    (150, 39.80, NOW() - interval '49 days');
