-- --- 1. Events table --- 
DROP TABLE IF EXISTS events CASCADE;
DROP TYPE IF EXISTS event_type;

-- Event types
CREATE TYPE event_type AS ENUM ('impression', 'click');

-- Table
CREATE TABLE IF NOT EXISTS events (
    app_id SERIAL PRIMARY KEY,
    event_type event_type NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Values
-- Values
INSERT INTO events (app_id, event_type, timestamp) VALUES
(1, 'impression', '2020-02-25 16:00:00'),
(2, 'click',      '2021-05-15 14:01:00'),
(3, 'impression', '2020-07-22 15:00:00'),
(4, 'click',      '2021-06-10 17:01:00'),
(5, 'impression', '2021-01-07 08:00:00'),
(6, 'click',      '2020-05-12 19:01:00'),
(7, 'click',      '2021-03-09 13:03:00'),
(8, 'impression', '2021-01-05 10:00:00'),
(9, 'click',      '2021-06-01 09:02:00'),
(10, 'impression', '2021-02-02 08:05:00'),
(11, 'impression', '2021-07-20 11:45:00'),
(12, 'impression', '2021-02-05 12:00:00'),
(13, 'impression', '2020-04-22 13:00:00'),
(14, 'click',      '2020-01-15 11:01:00'),
(15, 'impression', '2020-03-14 09:00:00'),
(16, 'click',      '2020-03-12 12:06:00'),
(17, 'click',      '2020-04-22 13:01:00'),
(18, 'click',      '2021-01-01 10:01:00'),
(19, 'impression', '2021-07-01 20:00:00'),
(20, 'click',      '2021-02-02 08:06:00'),
(21, 'impression', '2020-05-17 14:00:00'),
(22, 'impression', '2020-01-07 08:02:00'),
(23, 'impression', '2020-03-12 12:05:00'),
(24, 'impression', '2020-01-15 11:00:00'),
(25, 'click',      '2021-02-05 12:01:00'),
(26, 'impression', '2021-04-25 15:00:00'),
(27, 'click',      '2021-01-07 08:03:00'),
(28, 'click',      '2020-02-10 09:31:00'),
(29, 'click',      '2020-02-25 16:01:00'),
(30, 'click',      '2020-04-25 15:02:00'),
(31, 'impression', '2021-06-01 09:00:00'),
(32, 'impression', '2021-05-15 14:00:00'),
(33, 'impression', '2020-01-01 10:00:00'),
(34, 'impression', '2021-05-12 19:00:00'),
(35, 'impression', '2021-06-10 17:00:00'),
(36, 'impression', '2021-02-10 09:30:00'),
(37, 'impression', '2020-02-18 10:20:00'),
(38, 'impression', '2021-03-09 13:02:00'),
(39, 'click',      '2021-03-30 18:02:00'),
(40, 'impression', '2021-02-02 08:00:00'),
(41, 'click',      '2020-01-05 10:01:00'),
(42, 'impression', '2021-02-18 10:15:00'),
(43, 'impression', '2021-03-09 13:00:00'),
(44, 'impression', '2020-03-30 18:00:00'),
(45, 'click',      '2021-03-14 09:01:00'),
(46, 'impression', '2021-03-12 12:00:00'),
(47, 'click',      '2021-02-18 10:21:00'),
(48, 'click',      '2021-05-17 14:01:00'),
(49, 'impression', '2020-03-30 18:01:00'),
(50, 'impression', '2021-04-25 15:01:00');


