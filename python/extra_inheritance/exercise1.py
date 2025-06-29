

###----------------------- Exercise 1 ----------------------###

### Create a class Person with attributes name and age, and a class Student that inherits from Person and adds a grade attribute. 
# Add a method info() to print all the data.

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Student(Person):
    def __init__ (self, name, age, grade):
        super().__init__(name, age)
        self.grade = grade

    def info(self):
        print(f"The name is: {self.name}")
        print(f"The age name is: {self.age}")
        print(f"The grade name is: {self.grade}")

s = Student("Alice", 25 , "A")
s.info()
