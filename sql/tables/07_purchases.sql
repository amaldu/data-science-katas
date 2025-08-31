-- --- 7. Purchases table --- 

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS purchases (
    purchase_id INT PRIMARY KEY NOT NULL,
    user_id UUID DEFAULT gen_random_uuid(),
    product_id INT NOT NULL,
    price NUMERIC(4,2) NOT NULL,
    purchase_time TIMESTAMP NOT NULL
);

-- Values
INSERT INTO purchases (purchase_id, user_id, product_id, price, purchase_time)
VALUES
    (1, gen_random_uuid(), 101, 49.99, NOW() - interval '8 days'),
    (2, gen_random_uuid(), 102, 25.50, NOW() - interval '7 days'),
    (3, gen_random_uuid(), 103, 15.75, NOW() - interval '6 days'),
    (4, gen_random_uuid(), 104, 99.99, NOW() - interval '5 days'),
    (5, gen_random_uuid(), 101, 45.00, NOW() - interval '4 days'),
    (6, gen_random_uuid(), 102, 30.25, NOW() - interval '3 days'),
    (7, gen_random_uuid(), 103, 12.50, NOW() - interval '2 days'),
    (8, gen_random_uuid(), 104, 60.00, NOW() - interval '1 day'),
    (9, gen_random_uuid(), 101, 20.00, NOW() - interval '8 days'),
    (10, gen_random_uuid(), 102, 75.50, NOW() - interval '7 days'),
    (11, gen_random_uuid(), 103, 40.00, NOW() - interval '6 days'),
    (12, gen_random_uuid(), 104, 55.25, NOW() - interval '5 days'),
    (13, gen_random_uuid(), 101, 33.33, NOW() - interval '4 days'),
    (14, gen_random_uuid(), 102, 80.00, NOW() - interval '3 days'),
    (15, gen_random_uuid(), 103, 22.50, NOW() - interval '2 days'),
    (16, gen_random_uuid(), 104, 90.00, NOW() - interval '1 day'),
    (17, gen_random_uuid(), 101, 44.44, NOW() - interval '8 days'),
    (18, gen_random_uuid(), 102, 66.66, NOW() - interval '7 days'),
    (19, gen_random_uuid(), 103, 77.77, NOW() - interval '6 days'),
    (20, gen_random_uuid(), 104, 88.88, NOW());
