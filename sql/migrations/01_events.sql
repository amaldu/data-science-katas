-- --- 1. Events table --- 
DROP TABLE IF EXISTS events CASCADE;
DROP TYPE IF EXISTS event_type;

-- Event types
CREATE TYPE event_type AS ENUM ('impression', 'click');

-- Table
CREATE TABLE IF NOT EXISTS events (
    app_id INT NOT NULL,
    event_type event_type NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Values
INSERT INTO events (app_id, event_type, timestamp) VALUES
(7, 'impression', '2020-02-25 16:00:00'),
(2, 'click',      '2020-05-15 14:01:00'),
(9, 'impression', '2020-07-22 15:00:00'),
(4, 'click',      '2021-06-10 17:01:00'),
(5, 'impression', '2021-01-07 08:00:00'),
(6, 'click',      '2020-05-12 19:01:00'),
(1, 'click',      '2021-03-09 13:03:00'),
(8, 'impression', '2021-01-05 10:00:00'),
(3, 'click',      '2021-06-01 09:02:00'),
(9, 'impression', '2021-02-02 08:05:00'),
(2, 'impression', '2021-07-20 11:45:00'),
(7, 'impression', '2021-02-05 12:00:00'),
(4, 'impression', '2021-04-22 13:00:00'),
(8, 'click',      '2020-01-15 11:01:00'),
(6, 'impression', '2020-03-14 09:00:00'),
(5, 'click',      '2020-03-12 12:06:00'),
(3, 'click',      '2020-04-22 13:01:00'),
(1, 'click',      '2020-01-01 10:01:00'),
(9, 'impression', '2021-07-01 20:00:00'),
(2, 'click',      '2020-02-02 08:06:00'),
(7, 'impression', '2021-05-17 14:00:00'),
(4, 'impression', '2021-01-07 08:02:00'),
(1, 'impression', '2021-03-12 12:05:00'),
(5, 'impression', '2021-01-15 11:00:00'),
(8, 'click',      '2021-02-05 12:01:00'),
(6, 'impression', '2021-04-25 15:00:00'),
(3, 'click',      '2020-01-07 08:03:00'),
(9, 'click',      '2020-02-10 09:31:00'),
(7, 'click',      '2020-02-25 16:01:00'),
(2, 'click',      '2020-04-25 15:02:00'),
(4, 'impression', '2021-06-01 09:00:00'),
(1, 'impression', '2021-05-15 14:00:00'),
(8, 'impression', '2021-01-01 10:00:00'),
(5, 'impression', '2021-05-12 19:00:00'),
(9, 'impression', '2021-06-10 17:00:00'),
(6, 'impression', '2021-02-10 09:30:00'),
(3, 'impression', '2021-02-18 10:20:00'),
(7, 'impression', '2021-03-09 13:02:00'),
(2, 'click',      '2020-03-30 18:02:00'),
(4, 'impression', '2021-02-02 08:00:00'),
(1, 'click',      '2020-01-05 10:01:00'),
(8, 'impression', '2021-02-18 10:15:00'),
(5, 'impression', '2021-03-09 13:00:00'),
(9, 'impression', '2021-03-30 18:00:00'),
(6, 'click',      '2021-03-14 09:01:00'),
(3, 'impression', '2021-03-12 12:00:00'),
(7, 'click',      '2021-02-18 10:21:00'),
(2, 'click',      '2020-05-17 14:01:00'),
(4, 'impression', '2020-03-30 18:01:00'),
(1, 'impression', '2021-04-25 15:01:00');
