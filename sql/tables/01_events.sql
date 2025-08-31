-- --- 1. Events table --- 

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Event types
CREATE TYPE event_type AS ENUM ('impression', 'click');

-- Table
CREATE TABLE IF NOT EXISTS events (
    app_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id event_type NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Values
INSERT INTO events (event_id, timestamp)
VALUES
    ('impression', NOW() - interval '5 days'),
    ('click', NOW() - interval '4 days'),
    ('click', NOW() - interval '3 days'),
    ('impression', NOW() - interval '2 days'),
    ('click', NOW() - interval '1 day'),
    ('impression', NOW());

