-- --- 9. User transactions table --- 
DROP TABLE IF EXISTS user_transactions;

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- Table
CREATE TABLE user_transactions (
    transaction_id SERIAL PRIMARY KEY,
    product_id INT NOT NULL,
    user_id UUID NOT NULL DEFAULT gen_random_uuid(),
    spend FLOAT NOT NULL,
    transaction_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO user_transactions (product_id, spend, transaction_date)
VALUES
(101, 49.99, NOW() - interval '50 days'),
(102, 25.50, NOW() - interval '49 days'),
(103, 15.75, NOW() - interval '48 days'),
(104, 99.99, NOW() - interval '47 days'),
(105, 45.00, NOW() - interval '46 days'),
(106, 30.25, NOW() - interval '45 days'),
(107, 12.50, NOW() - interval '44 days'),
(108, 60.00, NOW() - interval '43 days'),
(109, 20.00, NOW() - interval '42 days'),
(110, 75.50, NOW() - interval '41 days'),
(111, 40.00, NOW() - interval '40 days'),
(112, 55.25, NOW() - interval '39 days'),
(113, 33.33, NOW() - interval '38 days'),
(114, 80.00, NOW() - interval '37 days'),
(115, 22.50, NOW() - interval '36 days'),
(116, 90.00, NOW() - interval '35 days'),
(117, 44.44, NOW() - interval '34 days'),
(118, 66.66, NOW() - interval '33 days'),
(119, 77.77, NOW() - interval '32 days'),
(120, 88.88, NOW() - interval '31 days'),
(121, 12.99, NOW() - interval '30 days'),
(122, 25.75, NOW() - interval '29 days'),
(123, 35.50, NOW() - interval '28 days'),
(124, 45.25, NOW() - interval '27 days'),
(125, 55.00, NOW() - interval '26 days'),
(126, 65.50, NOW() - interval '25 days'),
(127, 75.00, NOW() - interval '24 days'),
(128, 85.25, NOW() - interval '23 days'),
(129, 95.50, NOW() - interval '22 days'),
(130, 15.25, NOW() - interval '21 days'),
(131, 25.00, NOW() - interval '20 days'),
(132, 35.75, NOW() - interval '19 days'),
(133, 45.50, NOW() - interval '18 days'),
(134, 55.25, NOW() - interval '17 days'),
(135, 65.00, NOW() - interval '16 days'),
(136, 75.75, NOW() - interval '15 days'),
(137, 85.50, NOW() - interval '14 days'),
(138, 95.25, NOW() - interval '13 days'),
(139, 20.50, NOW() - interval '12 days'),
(140, 30.25, NOW() - interval '11 days'),
(141, 40.00, NOW() - interval '10 days'),
(142, 50.75, NOW() - interval '9 days'),
(143, 60.50, NOW() - interval '8 days'),
(144, 70.25, NOW() - interval '7 days'),
(145, 80.00, NOW() - interval '6 days'),
(146, 90.75, NOW() - interval '5 days'),
(147, 100.50, NOW() - interval '4 days'),
(148, 110.25, NOW() - interval '3 days'),
(149, 120.00, NOW() - interval '2 days'),
(150, 130.75, NOW() - interval '1 day');
