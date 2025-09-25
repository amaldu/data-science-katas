-- --- 7. Purchases table --- 
DROP TABLE IF EXISTS purchases;

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS purchases (
    purchase_id INT PRIMARY KEY NOT NULL,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    price NUMERIC(4,2) NOT NULL,
    purchase_time TIMESTAMP NOT NULL
);
-- Values
INSERT INTO purchases (purchase_id, user_id, product_id, price, purchase_time)
VALUES
    (1, 1, 101, 49.99, NOW() - interval '50 days'),
    (2, 4, 102, 25.50, NOW() - interval '49 days'),
    (3, 7, 103, 15.75, NOW() - interval '48 days'),
    (4, 4, 104, 99.99, NOW() - interval '47 days'),
    (5, 5, 101, 45.00, NOW() - interval '46 days'),
    (6, 5, 102, 30.25, NOW() - interval '45 days'),
    (7, 7, 103, 12.50, NOW() - interval '44 days'),
    (8, 8, 104, 60.00, NOW() - interval '43 days'),
    (9, 1, 101, 20.00, NOW() - interval '42 days'),
    (10, 7, 102, 75.50, NOW() - interval '41 days'),
    (11, 3, 103, 40.00, NOW() - interval '40 days'),
    (12, 5, 104, 55.25, NOW() - interval '39 days'),
    (13, 5, 101, 33.33, NOW() - interval '38 days'),
    (14, 7, 102, 80.00, NOW() - interval '37 days'),
    (15, 7, 103, 22.50, NOW() - interval '36 days'),
    (16, 4, 104, 90.00, NOW() - interval '35 days'),
    (17, 5, 101, 44.44, NOW() - interval '34 days'),
    (18, 7, 102, 66.66, NOW() - interval '33 days'),
    (19, 7, 103, 77.77, NOW() - interval '32 days'),
    (20, 4, 104, 88.88, NOW() - interval '31 days'),
    (21, 5, 101, 55.55, NOW() - interval '30 days'),
    (22, 6, 102, 23.45, NOW() - interval '29 days'),
    (23, 7, 103, 67.89, NOW() - interval '28 days'),
    (24, 8, 104, 34.56, NOW() - interval '27 days'),
    (25, 5, 101, 78.90, NOW() - interval '26 days'),
    (26, 2, 102, 45.67, NOW() - interval '25 days'),
    (27, 3, 103, 56.78, NOW() - interval '24 days'),
    (28, 4, 104, 89.12, NOW() - interval '23 days'),
    (29, 5, 101, 12.34, NOW() - interval '22 days'),
    (30, 5, 102, 34.56, NOW() - interval '21 days'),
    (31, 4, 103, 78.90, NOW() - interval '20 days'),
    (32, 8, 104, 23.45, NOW() - interval '19 days'),
    (33, 1, 101, 67.89, NOW() - interval '18 days'),
    (34, 4, 102, 45.12, NOW() - interval '17 days'),
    (35, 5, 103, 56.34, NOW() - interval '16 days'),
    (36, 4, 104, 88.76, NOW() - interval '15 days'),
    (37, 5, 101, 22.22, NOW() - interval '14 days'),
    (38, 4, 102, 33.33, NOW() - interval '13 days'),
    (39, 5, 103, 44.44, NOW() - interval '12 days'),
    (40, 8, 104, 55.55, NOW() - interval '11 days'),
    (41, 1, 101, 66.66, NOW() - interval '10 days'),
    (42, 2, 102, 77.77, NOW() - interval '9 days'),
    (43, 3, 103, 88.88, NOW() - interval '8 days'),
    (44, 4, 104, 99.99, NOW() - interval '7 days'),
    (45, 4, 101, 11.11, NOW() - interval '6 days'),
    (46, 4, 102, 22.22, NOW() - interval '5 days'),
    (47, 7, 103, 33.33, NOW() - interval '4 days'),
    (48, 8, 104, 44.44, NOW() - interval '3 days'),
    (49, 7, 101, 55.55, NOW() - interval '2 days'),
    (50, 2, 102, 66.66, NOW() - interval '1 day');

