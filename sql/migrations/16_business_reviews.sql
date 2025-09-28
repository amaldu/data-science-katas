-- --- Business review table ---
DROP TABLE IF EXISTS business_reviews;

-- Table
CREATE TABLE IF NOT EXISTS business_reviews (
    business_id INT NOT NULL,
    user_id INT NOT NULL,
    review_text VARCHAR(100) NOT NULL,
    review_stars INT NOT NULL,
    review_date TIMESTAMP NOT NULL
);

-- Values
INSERT INTO business_reviews (business_id, user_id, review_text, review_stars, review_date)
VALUES
(103, 3, 'Average experience, nothing special.', 3, NOW() - interval '48 days'),
(104, 4, 'Not satisfied, delivery was late.', 2, NOW() - interval '47 days'),
(105, 5, 'Terrible, would not buy again.', 1, NOW() - interval '46 days'),
(106, 6, 'Great value for money!', 5, NOW() - interval '45 days'),
(101, 1, 'Customer service was excellent.', 5, NOW() - interval '44 days'),
(102, 2, 'Product meets expectations.', 4, NOW() - interval '43 days'),
(103, 3, 'Fast shipping, very happy.', 5, NOW() - interval '42 days'),
(104, 4, 'Quality could be better.', 3, NOW() - interval '41 days'),
(105, 5, 'Not as described.', 2, NOW() - interval '40 days'),
(106, 6, 'Loved it!', 5, NOW() - interval '39 days'),
(101, 1, 'Good, but packaging was poor.', 4, NOW() - interval '38 days'),
(102, 2, 'Excellent purchase!', 5, NOW() - interval '37 days'),
(103, 3, 'Average, nothing to complain.', 3, NOW() - interval '36 days'),
(104, 4, 'Late delivery, not happy.', 2, NOW() - interval '35 days'),
(105, 5, 'Terrible experience.', 1, NOW() - interval '34 days'),
(106, 6, 'Worth every penny!', 5, NOW() - interval '33 days'),
(101, 1, 'Customer support was great.', 5, NOW() - interval '32 days'),
(102, 2, 'Product is okay.', 4, NOW() - interval '31 days'),
(103, 3, 'Satisfied with the purchase.', 4, NOW() - interval '30 days'),
(104, 4, 'Could improve quality.', 3, NOW() - interval '29 days'),
(105, 5, 'Not recommended.', 2, NOW() - interval '28 days'),
(106, 6, 'Amazing!', 5, NOW() - interval '27 days'),
(101, 1, 'Fast delivery.', 5, NOW() - interval '26 days'),
(102, 2, 'Product okay, nothing special.', 3, NOW() - interval '25 days'),
(103, 3, 'Highly recommend!', 5, NOW() - interval '24 days'),
(104, 4, 'Not satisfied.', 2, NOW() - interval '23 days'),
(105, 5, 'Terrible quality.', 1, NOW() - interval '22 days'),
(106, 6, 'Very happy!', 5, NOW() - interval '21 days'),
(101, 1, 'Good value.', 4, NOW() - interval '20 days'),
(102, 2, 'Okay purchase.', 3, NOW() - interval '19 days'),
(103, 3, 'Excellent!', 5, NOW() - interval '18 days'),
(104, 4, 'Could be better.', 3, NOW() - interval '17 days'),
(105, 5, 'Not satisfied.', 2, NOW() - interval '16 days'),
(106, 6, 'Amazing product!', 5, NOW() - interval '15 days'),
(101, 1, 'Very good!', 5, NOW() - interval '14 days'),
(102, 2, 'Average.', 3, NOW() - interval '13 days'),
(103, 3, 'Highly satisfied!', 5, NOW() - interval '12 days'),
(104, 4, 'Late delivery.', 2, NOW() - interval '11 days'),
(105, 5, 'Do not buy.', 1, NOW() - interval '10 days'),
(106, 6, 'Excellent quality.', 5, NOW() - interval '9 days'),
(101, 1, 'Great!', 5, NOW() - interval '8 days'),
(102, 2, 'Okay product.', 3, NOW() - interval '7 days'),
(103, 3, 'Very happy!', 5, NOW() - interval '6 days'),
(104, 4, 'Not impressed.', 2, NOW() - interval '5 days'),
(105, 5, 'Bad quality.', 1, NOW() - interval '4 days'),
(106, 6, 'Excellent!', 5, NOW() - interval '3 days'),
(101, 1, 'Good!', 4, NOW() - interval '2 days'),
(102, 2, 'Okay.', 3, NOW() - interval '1 day');
