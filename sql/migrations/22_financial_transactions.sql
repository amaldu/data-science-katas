-- --- Financial Transactions table ---
DROP TABLE IF EXISTS financial_transactions;

-- Table
CREATE TABLE IF NOT EXISTS financial_transactions (
    transaction_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    amount FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO financial_transactions (transaction_id, user_id, amount, transaction_date)
VALUES
(1, 1, 100.50, NOW() - interval '50 days'),
(2, 2, 200.25, NOW() - interval '49 days'),
(3, 3, 150.75, NOW() - interval '48 days'),
(4, 4, 300.00, NOW() - interval '47 days'),
(5, 5, 250.33, NOW() - interval '46 days'),
(6, 6, 175.50, NOW() - interval '45 days'),
(7, 1, 220.44, NOW() - interval '44 days'),
(8, 2, 110.25, NOW() - interval '43 days'),
(9, 3, 90.50, NOW() - interval '42 days'),
(10, 4, 130.75, NOW() - interval '41 days'),
(11, 5, 140.50, NOW() - interval '40 days'),
(12, 6, 160.25, NOW() - interval '39 days'),
(13, 1, 180.00, NOW() - interval '38 days'),
(14, 2, 190.25, NOW() - interval '37 days'),
(15, 3, 200.50, NOW() - interval '36 days'),
(16, 4, 210.75, NOW() - interval '35 days'),
(17, 5, 220.00, NOW() - interval '34 days'),
(18, 6, 230.25, NOW() - interval '33 days'),
(19, 1, 240.50, NOW() - interval '32 days'),
(20, 2, 250.75, NOW() - interval '31 days'),
(21, 3, 260.00, NOW() - interval '30 days'),
(22, 4, 270.25, NOW() - interval '29 days'),
(23, 5, 280.50, NOW() - interval '28 days'),
(24, 6, 290.75, NOW() - interval '27 days'),
(25, 1, 300.00, NOW() - interval '26 days'),
(26, 2, 310.25, NOW() - interval '25 days'),
(27, 3, 320.50, NOW() - interval '24 days'),
(28, 4, 330.75, NOW() - interval '3 days'),
(29, 5, 340.00, NOW() - interval '2 days'),
(30, 6, 350.25, NOW() - interval '1 days'),
(31, 1, 360.50, NOW() - interval '20 days'),
(32, 2, 370.75, NOW() - interval '19 days'),
(33, 3, 380.00, NOW() - interval '8 days'),
(34, 4, 390.25, NOW() - interval '7 days'),
(35, 5, 400.50, NOW() - interval '6 days'),
(36, 6, 410.75, NOW() - interval '5 days'),
(37, 1, 420.00, NOW() - interval '4 days'),
(38, 2, 430.25, NOW() - interval '3 days'),
(39, 3, 440.50, NOW() - interval '2 days'),
(40, 4, 450.75, NOW() - interval '1 days'),
(41, 5, 460.00, NOW() - interval '10 days'),
(42, 6, 470.25, NOW() - interval '9 days'),
(43, 1, 480.50, NOW() - interval '8 days'),
(44, 2, 490.75, NOW() - interval '7 days'),
(45, 3, 500.00, NOW() - interval '6 days'),
(46, 4, 510.25, NOW() - interval '5 days'),
(47, 5, 520.50, NOW() - interval '4 days'),
(48, 6, 530.75, NOW() - interval '3 days'),
(49, 1, 540.00, NOW() - interval '2 days'),
(50, 2, 550.25, NOW() - interval '1 day');