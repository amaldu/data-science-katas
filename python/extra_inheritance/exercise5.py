


###----------------------- Exercise 5 ----------------------###

# Create two classes: Flyer with a method fly() that returns "flying" and Walker with a method walk() that returns "walking". 
# Then create a class Bird that inherits from both and call both methods from an instance.

class Flyer:
    def fly(self):
        print("flying")

class Walker:
    def walk(self):
        print("walking")


class Bird(Flyer, Walker):
    pass


b = Bird()
b.fly()
b.walk()