



###----------------------- Exercise 4 ----------------------###

# Create a class Vehicle with attributes brand and model. 
# Then create a subclass Car that adds a year attribute and a method description() to print all the data.

class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
        pass

class Car(Vehicle):
    def __init__(self, brand, model, year):
        super().__init__(brand, model)
        self.year = year

    def description(self):
        print(f"The brand is: {self.brand}")
        print(f"The model name is: {self.model}")
        print(f"The year name is: {self.year}")

car = Car("Toyota", "Corolla", 2022)
car.description()
