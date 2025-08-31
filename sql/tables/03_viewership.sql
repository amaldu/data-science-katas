-- --- 3. Viewership table --- 

-- Table
CREATE TABLE IF NOT EXISTS viewership (
    user_id INT PRIMARY KEY, 
    device_type VARCHAR(20) NOT NULL,  
    view_time TIMESTAMP NOT NULL
);

-- Values
INSERT INTO viewership (user_id, device_type, view_time)
VALUES
    (101, 'mobile', NOW() - interval '5 days'),
    (102, 'desktop', NOW() - interval '4 days'),
    (103, 'tablet', NOW() - interval '3 days'),
    (104, 'mobile', NOW() - interval '2 days'),
    (105, 'desktop', NOW() - interval '1 day'),
    (106, 'tablet', NOW()),
    (101, 'desktop', NOW() - interval '4 days 3 hours'),
    (102, 'mobile', NOW() - interval '3 days 5 hours'),
    (103, 'desktop', NOW() - interval '2 days 2 hours'),
    (104, 'tablet', NOW() - interval '1 day 4 hours'),
    (105, 'mobile', NOW() - interval '5 hours'),
    (106, 'desktop', NOW() - interval '3 hours');
