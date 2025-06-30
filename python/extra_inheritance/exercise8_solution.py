



###----------------------- Exercise 8 ----------------------###

# Create a class LivingThing.
# Create a subclass Animal, and another subclass Bird that inherits from Animal.
# Each class should have a method describe() that returns a string identifying the level of the class.

class LivingThing:
    def describe(self):
        return f"I'm a living thing"


class Animal(LivingThing):
    def describe(self):
        return "I'm a animal"

class Bird(Animal):
    def describe(self):
        return "I'm a biiird"

thing = LivingThing()
animal = Animal()
bird = Bird()

print(thing.describe())  
print(animal.describe()) 
print(bird.describe())  
