-- --- Supermarket Products table ---
DROP TABLE IF EXISTS supermarket_products;

-- Table
CREATE TABLE IF NOT EXISTS supermarket_products (
    product_id INT,
    product_name VARCHAR(50) NOT NULL,
    price FLOAT NOT NULL
);

-- Values
INSERT INTO supermarket_products (product_id, product_name, price)
VALUES
(101, 'Apple', 1.25),
(102, 'Banana', 0.75),
(103, 'Orange', 1.50),
(104, 'Milk', 2.20),
(105, 'Bread', 1.80),
(106, 'Eggs', 3.00),
(107, 'Cheese', 4.50),
(108, 'Tomato', 1.10),
(109, 'Potato', 0.90),
(110, 'Chicken', 5.75);
