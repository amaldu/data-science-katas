-- --- 2. Trades table --- 
DROP TABLE IF EXISTS trades CASCADE;
DROP TYPE IF EXISTS status_type CASCADE;

-- Trade types
CREATE TYPE status_type AS ENUM ('complete', 'cancelled');

-- Table
CREATE TABLE IF NOT EXISTS trades (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    price NUMERIC(6, 2) NOT NULL,  
    quantity INT NOT NULL, 
    status status_type NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Values
INSERT INTO trades (order_id, user_id, price, quantity, status, timestamp)
VALUES
    (1, 101, 49.99, 10, 'complete', NOW() - interval '50 days'),
    (2, 102, 75.50, 5, 'complete', NOW() - interval '49 days'),
    (3, 102, 20.00, 15, 'cancelled', NOW() - interval '48 days'),
    (4, 104, 100.00, 8, 'complete', NOW() - interval '47 days'),
    (5, 102, 60.25, 20, 'cancelled', NOW() - interval '46 days'),
    (6, 106, 33.33, 12, 'complete', NOW() - interval '45 days'),
    (7, 101, 85.75, 7, 'cancelled', NOW() - interval '44 days'),
    (8, 102, 42.10, 18, 'complete', NOW() - interval '43 days'),
    (9, 103, 55.55, 9, 'complete', NOW() - interval '42 days'),
    (10, 104, 99.99, 11, 'cancelled', NOW() - interval '41 days'),
    (11, 105, 44.44, 14, 'cancelled', NOW() - interval '40 days'),
    (12, 106, 77.77, 6, 'complete', NOW() - interval '39 days'),
    (13, 101, 23.45, 13, 'complete', NOW() - interval '38 days'),
    (14, 102, 88.88, 5, 'complete', NOW() - interval '37 days'),
    (15, 103, 66.66, 17, 'cancelled', NOW() - interval '36 days'),
    (16, 104, 31.11, 19, 'cancelled', NOW() - interval '35 days'),
    (17, 105, 50.50, 8, 'complete', NOW() - interval '34 days'),
    (18, 106, 45.45, 12, 'cancelled', NOW() - interval '33 days'),
    (19, 101, 72.25, 10, 'complete', NOW() - interval '32 days'),
    (20, 102, 95.50, 7, 'complete', NOW() - interval '31 days'),
    (21, 103, 30.30, 15, 'cancelled', NOW() - interval '30 days'),
    (22, 102, 60.60, 9, 'complete', NOW() - interval '29 days'),
    (23, 105, 80.80, 12, 'complete', NOW() - interval '28 days'),
    (24, 106, 55.55, 11, 'complete', NOW() - interval '27 days'),
    (25, 101, 44.44, 14, 'complete', NOW() - interval '26 days'),
    (26, 101, 77.77, 6, 'cancelled', NOW() - interval '25 days'),
    (27, 102, 33.33, 18, 'cancelled', NOW() - interval '24 days'),
    (28, 104, 99.99, 5, 'cancelled', NOW() - interval '23 days'),
    (29, 101, 50.50, 16, 'complete', NOW() - interval '22 days'),
    (30, 106, 65.65, 8, 'complete', NOW() - interval '21 days'),
    (31, 101, 20.20, 13, 'cancelled', NOW() - interval '20 days'),
    (32, 102, 75.75, 9, 'complete', NOW() - interval '19 days'),
    (33, 103, 88.88, 12, 'complete', NOW() - interval '18 days'),
    (34, 104, 45.45, 10, 'complete', NOW() - interval '17 days'),
    (35, 105, 55.55, 14, 'cancelled', NOW() - interval '16 days'),
    (36, 106, 66.66, 7, 'cancelled', NOW() - interval '15 days'),
    (37, 101, 77.77, 11, 'cancelled', NOW() - interval '14 days'),
    (38, 101, 88.88, 9, 'complete', NOW() - interval '13 days'),
    (39, 103, 99.99, 13, 'complete', NOW() - interval '12 days'),
    (40, 104, 22.22, 15, 'complete', NOW() - interval '11 days'),
    (41, 101, 33.33, 8, 'complete', NOW() - interval '10 days'),
    (42, 106, 44.44, 12, 'cancelled', NOW() - interval '9 days'),
    (43, 101, 55.55, 7, 'complete', NOW() - interval '8 days'),
    (44, 102, 66.66, 10, 'complete', NOW() - interval '7 days'),
    (45, 103, 77.77, 9, 'complete', NOW() - interval '6 days'),
    (46, 104, 88.88, 14, 'complete', NOW() - interval '5 days'),
    (47, 105, 99.99, 11, 'cancelled', NOW() - interval '4 days'),
    (48, 106, 20.20, 13, 'cancelled', NOW() - interval '3 days'),
    (49, 101, 30.30, 12, 'cancelled', NOW() - interval '2 days'),
    (50, 102, 40.40, 15, 'complete', NOW() - interval '1 day');
