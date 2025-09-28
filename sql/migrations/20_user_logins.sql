-- --- User Logins table ---
DROP TABLE IF EXISTS user_logins;

-- Table
CREATE TABLE IF NOT EXISTS user_logins (
    user_id INT NOT NULL,
    login_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO user_logins (user_id, login_date)
VALUES
(1, NOW() - interval '50 days'),
(2, NOW() - interval '49 days'),
(3, NOW() - interval '48 days'),
(4, NOW() - interval '47 days'),
(5, NOW() - interval '46 days'),
(6, NOW() - interval '45 days'),
(1, NOW() - interval '44 days'),
(2, NOW() - interval '43 days'),
(3, NOW() - interval '42 days'),
(4, NOW() - interval '41 days'),
(5, NOW() - interval '40 days'),
(6, NOW() - interval '39 days'),
(1, NOW() - interval '38 days'),
(2, NOW() - interval '37 days'),
(3, NOW() - interval '36 days'),
(4, NOW() - interval '35 days'),
(5, NOW() - interval '34 days'),
(6, NOW() - interval '33 days'),
(1, NOW() - interval '32 days'),
(2, NOW() - interval '31 days'),
(3, NOW() - interval '30 days'),
(4, NOW() - interval '29 days'),
(5, NOW() - interval '28 days'),
(6, NOW() - interval '27 days'),
(1, NOW() - interval '26 days'),
(2, NOW() - interval '25 days'),
(3, NOW() - interval '24 days'),
(4, NOW() - interval '23 days'),
(5, NOW() - interval '22 days'),
(6, NOW() - interval '21 days'),
(1, NOW() - interval '20 days'),
(2, NOW() - interval '19 days'),
(3, NOW() - interval '18 days'),
(4, NOW() - interval '17 days'),
(5, NOW() - interval '16 days'),
(6, NOW() - interval '15 days'),
(1, NOW() - interval '14 days'),
(2, NOW() - interval '13 days'),
(3, NOW() - interval '12 days'),
(4, NOW() - interval '11 days'),
(5, NOW() - interval '10 days'),
(6, NOW() - interval '9 days'),
(1, NOW() - interval '8 days'),
(2, NOW() - interval '7 days'),
(3, NOW() - interval '6 days'),
(4, NOW() - interval '5 days'),
(5, NOW() - interval '4 days'),
(6, NOW() - interval '3 days'),
(1, NOW() - interval '2 days'),
(2, NOW() - interval '1 day');