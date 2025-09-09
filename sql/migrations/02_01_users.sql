-- --- 2. Users table --- 
DROP TABLE IF EXISTS users CASCADE;

-- Table
CREATE TABLE users (
    user_id INT PRIMARY KEY, 
    city VARCHAR(20) NOT NULL,  
    email VARCHAR(50) NOT NULL, 
    signup_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO users (user_id, city, email, signup_date) VALUES
(101, 'Oslo', 'user101@example.com', NOW() - interval '50 days'),
(102, 'Barcelona', 'user102@example.com', NOW() - interval '49 days'),
(103, 'Amsterdam', 'user103@example.com', NOW() - interval '48 days'),
(104, 'Sevilla', 'user104@example.com', NOW() - interval '47 days'),
(105, 'Berlin', 'user105@example.com', NOW() - interval '46 days'),
(106, 'Paris', 'user106@example.com', NOW() - interval '45 days');
