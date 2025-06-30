


###----------------------- Exercise 7 ----------------------###


# Create a class Employee with name and salary.
# Create a subclass Manager that adds an optional list of employees_managed and a method add_employee(emp).


class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        pass

class Manager(Employee):
    def __init__(self, name, salary, employees_managed = None):
        super().__init__(name,salary)
        if employees_managed is None:
            self.employees_managed = []
        else:
            self.employees_managed = employees_managed
    
    def add_employees(self, emp):
        if emp not in self.employees_managed:
            self.employees_managed.append(emp)

e1 = Employee("Tom", 50000)
e2 = Employee("Jerry", 52000)
m1 = Manager("Alice", 80000)
m1.add_employees(e1)
m1.add_employees(e2)
print([e.name for e in m1.employees_managed]) 
        
