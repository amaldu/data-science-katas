-- --- 8. Jobs listings table --- 
DROP TABLE IF EXISTS job_listings;

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table
CREATE TABLE IF NOT EXISTS job_listings (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id SERIAL NOT NULL,
    title VARCHAR(30) NOT NULL,
    description VARCHAR(150) NOT NULL,
    post_date TIMESTAMP NOT NULL
);

-- Values 
INSERT INTO job_listings (company_id, title, description, post_date)
VALUES
    (17, 'Project Manager', 'Coordinate projects and teams', NOW() - interval '34 days'),
    (2, 'Data Analyst', 'Analyze data and generate reports', NOW() - interval '49 days'),
    (32, 'QA Engineer', 'Test and ensure software quality', NOW() - interval '19 days'),
    (36, 'Product Manager', 'Lead product development and strategy', NOW() - interval '15 days'),
    (8, 'Marketing Specialist', 'Plan and execute marketing campaigns', NOW() - interval '43 days'),
    (23, 'Machine Learning Engineer', 'Build and deploy ML models', NOW() - interval '28 days'),
    (33, 'Data Analyst', 'Analyze business data and KPIs', NOW() - interval '18 days'),
    (41, 'Graphic Designer', 'Create visual content and branding', NOW() - interval '10 days'),
    (7, 'UX Designer', 'Design user interfaces and experiences', NOW() - interval '44 days'),
    (20, 'Data Engineer', 'Build and maintain data pipelines', NOW() - interval '31 days'),
    (16, 'Backend Developer', 'Develop server-side logic and APIs', NOW() - interval '35 days'),
    (49, 'Fullstack Developer', 'Develop both frontend and backend systems', NOW() - interval '2 days'),
    (42, 'Machine Learning Engineer', 'Build and deploy ML models', NOW() - interval '9 days'),
    (3, 'Software Engineer', 'Develop and maintain web applications', NOW() - interval '48 days'),
    (35, 'Software Engineer', 'Develop and maintain microservices', NOW() - interval '16 days'),
    (6, 'UX Designer', 'Design user interfaces and experiences', NOW() - interval '45 days'),
    (28, 'Technical Writer', 'Document software and technical processes', NOW() - interval '23 days'),
    (40, 'Frontend Developer', 'Develop client-side applications', NOW() - interval '11 days'),
    (1, 'Data Analyst', 'Analyze data and generate reports', NOW() - interval '50 days'),
    (50, 'Business Analyst', 'Analyze business processes and KPIs', NOW() - interval '1 day'),
    (10, 'Data Scientist', 'Build models and analyze datasets', NOW() - interval '41 days'),
    (14, 'Business Analyst', 'Analyze business processes and KPIs', NOW() - interval '37 days'),
    (46, 'Security Analyst', 'Ensure cybersecurity and compliance', NOW() - interval '5 days'),
    (11, 'DevOps Engineer', 'Manage infrastructure and deployments', NOW() - interval '40 days'),
    (48, 'Project Manager', 'Coordinate projects and teams', NOW() - interval '3 days'),
    (24, 'Business Intelligence Analyst', 'Generate insights from business data', NOW() - interval '27 days'),
    (31, 'DevOps Engineer', 'Manage infrastructure and deployments', NOW() - interval '20 days'),
    (5, 'Product Manager', 'Lead product development and strategy', NOW() - interval '46 days'),
    (25, 'Cloud Engineer', 'Maintain cloud infrastructure and services', NOW() - interval '26 days'),
    (30, 'Fullstack Developer', 'Develop both frontend and backend systems', NOW() - interval '21 days'),
    (43, 'Cloud Engineer', 'Maintain cloud infrastructure and services', NOW() - interval '8 days'),
    (19, 'Data Engineer', 'Build and maintain data pipelines', NOW() - interval '32 days'),
    (21, 'Software Architect', 'Design software architecture and patterns', NOW() - interval '30 days'),
    (9, 'Data Scientist', 'Build models and analyze datasets', NOW() - interval '42 days'),
    (18, 'Graphic Designer', 'Create visual content and branding', NOW() - interval '33 days'),
    (15, 'Frontend Developer', 'Develop client-side applications', NOW() - interval '36 days'),
    (26, 'Mobile Developer', 'Develop mobile applications for iOS and Android', NOW() - interval '25 days'),
    (27, 'Security Analyst', 'Ensure cybersecurity and compliance', NOW() - interval '24 days'),
    (13, 'HR Manager', 'Manage recruitment and employee relations', NOW() - interval '38 days'),
    (12, 'QA Engineer', 'Test and ensure software quality', NOW() - interval '39 days'),
    (22, 'Database Administrator', 'Manage and maintain database systems', NOW() - interval '29 days'),
    (44, 'Database Administrator', 'Manage and maintain database systems', NOW() - interval '7 days'),
    (29, 'Network Engineer', 'Manage networking infrastructure', NOW() - interval '22 days'),
    (45, 'Network Engineer', 'Manage networking infrastructure', NOW() - interval '6 days'),
    (4, 'Software Engineer', 'Develop and maintain web applications', NOW() - interval '47 days'),
    (37, 'Data Scientist', 'Analyze datasets and build predictive models', NOW() - interval '14 days'),
    (38, 'Marketing Specialist', 'Plan and execute marketing campaigns', NOW() - interval '13 days'),
    (34, 'UX Designer', 'Design user interfaces and experiences', NOW() - interval '17 days'),
    (39, 'Backend Developer', 'Develop server-side logic and APIs', NOW() - interval '12 days'),
    (46, 'Technical Writer', 'Document software and technical processes', NOW() - interval '4 days');
