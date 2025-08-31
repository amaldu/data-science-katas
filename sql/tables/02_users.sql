-- --- 2. Users table --- 

-- Table
CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY, 
    city VARCHAR(20) NOT NULL,  
    email VARCHAR(25) NOT NULL, 
    signup_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO users (user_id, city, email, signup_date)
VALUES
    (101, 'Oslo', 'user101@example.com', NOW() - interval '10 days'),
    (102, 'Barcelona', 'user102@example.com', NOW() - interval '8 days'),
    (103, 'Amssterdam', 'user103@example.com', NOW() - interval '15 days'),
    (104, 'Sevilla', 'user104@example.com', NOW() - interval '7 days'),
    (105, 'Berlin', 'user105@example.com', NOW() - interval '20 days');

