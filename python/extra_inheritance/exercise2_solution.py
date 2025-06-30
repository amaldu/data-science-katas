

###----------------------- Exercise 2 ----------------------###

# Create a class Animal with a method speak(). 
# Then create a class Dog that inherits from Animal and overrides speak() to say "Woof!".

class Animal:
    def speak(self):
        return "some sound"
    
class Dog(Animal):
    def speak(self):
        return "Woof!"
    

dog = Dog()
print(dog.speak())