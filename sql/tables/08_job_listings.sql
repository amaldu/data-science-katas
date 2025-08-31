-- --- 8. Jobs listings table --- 

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS job_listings (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id SERIAL NOT NULL,
    title VARCHAR(20) NOT NULL,
    description VARCHAR(150) NOT NULL,
    post_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO job_listings (company_id, title, description, post_date)
VALUES
    (1, 'Data Analyst', 'Analyze data and generate reports', NOW() - interval '10 days'),
    (1, 'Data Analyst', 'Analyze data and generate reports', NOW() - interval '9 days'),
    (2, 'Software Engineer', 'Develop and maintain web applications', NOW() - interval '8 days'),
    (2, 'Software Engineer', 'Develop and maintain web applications', NOW() - interval '7 days'),
    (3, 'Product Manager', 'Lead product development and strategy', NOW() - interval '6 days'),
    (4, 'UX Designer', 'Design user interfaces and experiences', NOW() - interval '5 days'),
    (4, 'UX Designer', 'Design user interfaces and experiences', NOW() - interval '4 days'),
    (5, 'Marketing Specialist', 'Plan and execute marketing campaigns', NOW() - interval '3 days'),
    (6, 'Data Scientist', 'Build models and analyze datasets', NOW() - interval '2 days'),
    (6, 'Data Scientist', 'Build models and analyze datasets', NOW() - interval '1 day'),
    (7, 'DevOps Engineer', 'Manage infrastructure and deployments', NOW() - interval '8 days'),
    (8, 'QA Engineer', 'Test and ensure software quality', NOW() - interval '7 days'),
    (9, 'HR Manager', 'Manage recruitment and employee relations', NOW() - interval '6 days'),
    (10, 'Business Analyst', 'Analyze business processes and KPIs', NOW() - interval '5 days'),
    (11, 'Frontend Developer', 'Develop client-side applications', NOW() - interval '4 days'),
    (12, 'Backend Developer', 'Develop server-side logic and APIs', NOW() - interval '3 days'),
    (13, 'Project Manager', 'Coordinate projects and teams', NOW() - interval '2 days'),
    (14, 'Graphic Designer', 'Create visual content and branding', NOW() - interval '1 day'),
    (15, 'Data Engineer', 'Build and maintain data pipelines', NOW() - interval '2 days'),
    (15, 'Data Engineer', 'Build and maintain data pipelines', NOW());

