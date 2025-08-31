-- --- 6. Tweets table --- 

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INT NOT NULL,
    msg VARCHAR(180) NOT NULL,
    tweet_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO tweets (user_id, msg, tweet_date)
VALUES
    (101, 'Loving the new data project! #datascience', '2020-03-15 10:30:00'),
    (102, 'Just finished a marathon, feeling great!', '2020-06-22 08:15:00'),
    (103, 'Coffee and code, my perfect morning routine.', '2020-09-05 07:50:00'),
    (104, 'Exploring machine learning with PostgreSQL', '2020-12-11 14:20:00'),
    (105, 'Watching the sunset, life is good.', '2020-11-01 18:45:00'),
    (106, 'Reading a new book on AI trends.', '2020-05-19 16:00:00'),
    (101, 'Debugging Python all day! #programming', '2021-01-10 09:30:00'),
    (102, 'Enjoying a sunny weekend at the park.', '2021-02-14 11:15:00'),
    (103, 'Just launched my personal website!', '2021-03-20 13:00:00'),
    (104, 'Learning SQL joins, fun and challenging.', '2021-04-25 15:40:00'),
    (105, 'Trying out a new coffee recipe today.', '2021-05-30 07:25:00'),
    (106, 'Participated in an online hackathon.', '2021-06-15 22:10:00');
    (101, 'Trying out a new Python library today!', '2020-07-02 12:10:00'),
    (102, 'Watching a great movie tonight.', '2020-08-16 20:45:00'),
    (103, 'Finally solved that tricky SQL query!', '2020-09-28 09:00:00'),
    (104, 'Attending an online data meetup.', '2020-10-05 17:30:00'),
    (105, 'Learning about Docker and containers.', '2020-11-12 14:50:00'),
    (106, 'Just finished a 5K run!', '2020-12-20 08:15:00'),
    (101, 'Trying to balance work and study.', '2021-01-18 13:45:00'),
    (102, 'Cooking a new recipe tonight.', '2021-02-05 19:30:00'),
    (103, 'Exploring new features in PostgreSQL.', '2021-03-11 10:25:00'),
    (104, 'Writing some notes on machine learning.', '2021-04-22 16:40:00'),
    (105, 'Attending a virtual conference today.', '2021-05-03 09:50:00'),
    (106, 'Reading a blog about AI ethics.', '2021-06-18 11:15:00'),
    (101, 'Morning coffee and coding session.', '2021-07-09 07:20:00'),
    (102, 'Weekend hiking adventure.', '2021-08-21 14:05:00'),
    (103, 'Exploring new Python frameworks.', '2021-09-30 15:45:00');

