


###----------------------- Exercise 3 ----------------------###

# Create a class Employee with a class attribute company = "ACME", and a subclass Manager. 
# Print the company name from an instance of Manager.


class Employee:
    company = "ACME"

class Manager(Employee):
    pass

man_1 = Manager()
print(man_1.company)