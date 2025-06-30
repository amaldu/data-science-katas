


###----------------------- Exercise 6 ----------------------###

# Create a class Vehicle with a method start() that prints "Starting vehicle".
# Create a subclass Car that overrides start() and prints "Starting car" after calling the parent method.

class Vehicle:
    def start(self):
        print("Starting Vehicle")

class Car(Vehicle):
    def start(self):
        super().start() 
        print("Starting car")


v = Vehicle()
v.start()


c = Car()
c.start()
