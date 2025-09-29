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
(1, NOW() - interval '90 days'),
(2, NOW() - interval '99 days'),
(3, NOW() - interval '98 days'),
(4, NOW() - interval '97 days'),
(5, NOW() - interval '96 days'),
(6, NOW() - interval '95 days'),
(1, NOW() - interval '94 days'),
(2, NOW() - interval '93 days'),
(3, NOW() - interval '92 days'),
(4, NOW() - interval '81 days'),
(5, NOW() - interval '80 days'),
(6, NOW() - interval '89 days'),
(1, NOW() - interval '88 days'),
(2, NOW() - interval '97 days'),
(3, NOW() - interval '96 days'),
(4, NOW() - interval '95 days'),
(5, NOW() - interval '94 days'),
(6, NOW() - interval '83 days'),
(1, NOW() - interval '82 days'),
(2, NOW() - interval '81 days'),
(3, NOW() - interval '80 days'),
(4, NOW() - interval '89 days'),
(5, NOW() - interval '78 days'),
(6, NOW() - interval '77 days'),
(1, NOW() - interval '76 days'),
(2, NOW() - interval '75 days'),
(5, NOW() - interval '44 days'),
(4, NOW() - interval '43 days'),
(5, NOW() - interval '42 days'),
(3, NOW() - interval '41 days'),
(1, NOW() - interval '40 days'),
(2, NOW() - interval '49 days'),
(3, NOW() - interval '48 days'),
(1, NOW() - interval '37 days'),
(5, NOW() - interval '36 days'),
(1, NOW() - interval '35 days'),
(1, NOW() - interval '34 days'),
(2, NOW() - interval '33 days'),
(2, NOW() - interval '12 days'),
(4, NOW() - interval '11 days'),
(5, NOW() - interval '10 days'),
(6, NOW() - interval '9 days'),
(1, NOW() - interval '8 days'),
(8, NOW() - interval '7 days'),
(2, NOW() - interval '6 days'),
(4, NOW() - interval '5 days'),
(2, NOW() - interval '4 days'),
(6, NOW() - interval '3 days'),
(9, NOW() - interval '2 days'),
(2, NOW() - interval '1 day');