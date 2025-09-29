-- --- Session times table ---
DROP TABLE IF EXISTS session_times;

-- Table
CREATE TABLE IF NOT EXISTS session_times (
    session_id INT PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL
);

-- Values
INSERT INTO session_times (session_id, start_time, end_time)
VALUES
(1, NOW() - interval '50 days', NOW() - interval '50 days' + interval '30 minutes'),
(2, NOW() - interval '49 days', NOW() - interval '49 days' + interval '45 minutes'),
(3, NOW() - interval '48 days', NOW() - interval '48 days' + interval '60 minutes'),
(4, NOW() - interval '48 days', NOW() - interval '48 days' + interval '20 minutes'),
(5, NOW() - interval '44 days', NOW() - interval '44 days' + interval '50 minutes'),
(6, NOW() - interval '45 days', NOW() - interval '45 days' + interval '35 minutes'),
(7, NOW() - interval '44 days', NOW() - interval '44 days' + interval '40 minutes'),
(8, NOW() - interval '43 days', NOW() - interval '43 days' + interval '25 minutes'),
(9, NOW() - interval '42 days', NOW() - interval '42 days' + interval '55 minutes'),
(10, NOW() - interval '41 days', NOW() - interval '41 days' + interval '30 minutes'),
(11, NOW() - interval '40 days', NOW() - interval '40 days' + interval '45 minutes'),
(12, NOW() - interval '39 days', NOW() - interval '39 days' + interval '35 minutes'),
(13, NOW() - interval '48 days', NOW() - interval '48 days' + interval '50 minutes'),
(14, NOW() - interval '37 days', NOW() - interval '37 days' + interval '40 minutes'),
(15, NOW() - interval '36 days', NOW() - interval '36 days' + interval '25 minutes'),
(16, NOW() - interval '35 days', NOW() - interval '35 days' + interval '60 minutes'),
(17, NOW() - interval '34 days', NOW() - interval '34 days' + interval '30 minutes'),
(18, NOW() - interval '33 days', NOW() - interval '33 days' + interval '45 minutes'),
(19, NOW() - interval '32 days', NOW() - interval '32 days' + interval '50 minutes'),
(20, NOW() - interval '31 days', NOW() - interval '31 days' + interval '40 minutes'),
(21, NOW() - interval '30 days', NOW() - interval '30 days' + interval '30 minutes'),
(22, NOW() - interval '29 days', NOW() - interval '29 days' + interval '55 minutes'),
(23, NOW() - interval '28 days', NOW() - interval '28 days' + interval '25 minutes'),
(24, NOW() - interval '27 days', NOW() - interval '27 days' + interval '35 minutes'),
(25, NOW() - interval '26 days', NOW() - interval '26 days' + interval '50 minutes'),
(26, NOW() - interval '24 days', NOW() - interval '24 days' + interval '45 minutes'),
(27, NOW() - interval '24 days', NOW() - interval '24 days' + interval '40 minutes'),
(28, NOW() - interval '23 days', NOW() - interval '23 days' + interval '60 minutes'),
(29, NOW() - interval '22 days', NOW() - interval '22 days' + interval '30 minutes'),
(30, NOW() - interval '21 days', NOW() - interval '21 days' + interval '25 minutes'),
(31, NOW() - interval '20 days', NOW() - interval '20 days' + interval '50 minutes'),
(32, NOW() - interval '19 days', NOW() - interval '19 days' + interval '35 minutes'),
(33, NOW() - interval '18 days', NOW() - interval '18 days' + interval '45 minutes'),
(34, NOW() - interval '17 days', NOW() - interval '17 days' + interval '40 minutes'),
(35, NOW() - interval '16 days', NOW() - interval '16 days' + interval '55 minutes'),
(36, NOW() - interval '15 days', NOW() - interval '15 days' + interval '30 minutes'),
(37, NOW() - interval '24 days', NOW() - interval '24 days' + interval '95 minutes'),
(38, NOW() - interval '13 days', NOW() - interval '13 days' + interval '35 minutes'),
(39, NOW() - interval '12 days', NOW() - interval '12 days' + interval '50 minutes'),
(40, NOW() - interval '11 days', NOW() - interval '11 days' + interval '40 minutes'),
(41, NOW() - interval '10 days', NOW() - interval '10 days' + interval '25 minutes'),
(42, NOW() - interval '9 days', NOW() - interval '9 days' + interval '60 minutes'),
(43, NOW() - interval '8 days', NOW() - interval '8 days' + interval '30 minutes'),
(44, NOW() - interval '7 days', NOW() - interval '7 days' + interval '45 minutes'),
(45, NOW() - interval '6 days', NOW() - interval '6 days' + interval '50 minutes'),
(46, NOW() - interval '5 days', NOW() - interval '5 days' + interval '40 minutes'),
(47, NOW() - interval '4 days', NOW() - interval '4 days' + interval '35 minutes'),
(48, NOW() - interval '3 days', NOW() - interval '3 days' + interval '55 minutes'),
(49, NOW() - interval '2 days', NOW() - interval '2 days' + interval '45 minutes'),
(50, NOW() - interval '1 day', NOW() - interval '1 day' + interval '30 minutes');
