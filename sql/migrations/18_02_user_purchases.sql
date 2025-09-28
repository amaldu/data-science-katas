-- --- User Purchases table ---
DROP TABLE IF EXISTS user_purchases;

-- Table
CREATE TABLE IF NOT EXISTS user_purchases (
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    purchase_amount FLOAT NOT NULL,
    purchase_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO user_purchases (user_id, product_id, purchase_amount, purchase_date)
VALUES
(1, 101, 25.50, NOW() - interval '50 days'),
(3, 102, 40.75, NOW() - interval '49 days'),
(5, 103, 15.25, NOW() - interval '48 days'),
(2, 104, 60.00, NOW() - interval '47 days'),
(6, 105, 33.33, NOW() - interval '46 days'),
(4, 106, 22.50, NOW() - interval '45 days'),
(1, 107, 44.44, NOW() - interval '44 days'),
(2, 108, 55.25, NOW() - interval '43 days'),
(5, 109, 12.50, NOW() - interval '42 days'),
(3, 110, 66.66, NOW() - interval '41 days'),
(6, 101, 75.50, NOW() - interval '40 days'),
(1, 102, 80.00, NOW() - interval '39 days'),
(2, 103, 30.25, NOW() - interval '38 days'),
(4, 104, 20.00, NOW() - interval '37 days'),
(5, 105, 45.75, NOW() - interval '36 days'),
(3, 106, 50.25, NOW() - interval '35 days'),
(6, 107, 60.50, NOW() - interval '34 days'),
(1, 108, 70.00, NOW() - interval '33 days'),
(2, 109, 80.25, NOW() - interval '32 days'),
(5, 110, 90.00, NOW() - interval '31 days'),
(3, 101, 12.75, NOW() - interval '30 days'),
(4, 102, 15.50, NOW() - interval '29 days'),
(6, 103, 18.25, NOW() - interval '28 days'),
(1, 104, 22.75, NOW() - interval '27 days'),
(2, 105, 33.00, NOW() - interval '26 days'),
(5, 106, 44.50, NOW() - interval '25 days'),
(3, 107, 55.75, NOW() - interval '24 days'),
(6, 108, 60.25, NOW() - interval '23 days'),
(1, 109, 70.50, NOW() - interval '22 days'),
(2, 110, 80.75, NOW() - interval '21 days'),
(4, 101, 90.25, NOW() - interval '20 days'),
(5, 102, 100.50, NOW() - interval '19 days'),
(3, 103, 110.75, NOW() - interval '18 days'),
(6, 104, 120.00, NOW() - interval '17 days'),
(1, 105, 12.50, NOW() - interval '16 days'),
(2, 106, 15.75, NOW() - interval '15 days'),
(5, 107, 18.00, NOW() - interval '14 days'),
(3, 108, 22.25, NOW() - interval '13 days'),
(6, 109, 30.50, NOW() - interval '12 days'),
(1, 110, 40.75, NOW() - interval '11 days'),
(2, 101, 50.25, NOW() - interval '10 days'),
(5, 102, 60.50, NOW() - interval '9 days'),
(3, 103, 70.75, NOW() - interval '8 days'),
(6, 104, 80.00, NOW() - interval '7 days'),
(1, 105, 90.25, NOW() - interval '6 days'),
(2, 106, 100.50, NOW() - interval '5 days'),
(5, 107, 110.75, NOW() - interval '4 days'),
(3, 108, 120.00, NOW() - interval '3 days'),
(6, 109, 12.25, NOW() - interval '2 days'),
(1, 110, 15.50, NOW() - interval '1 day');
