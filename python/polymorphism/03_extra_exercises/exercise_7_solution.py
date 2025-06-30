



###----------------------- Exercise 7 ----------------------###


# Define Circle and Square classes with a method .area() that returns the result of the area of each. 
# Then write a function print_area(shape) that prints the area. Use duck typing: the function must not care about the class.



class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        print(f" The area of the circle is : {3.14 * self.radius ** 2}")
    
class Square:
    def __init__(self,side):
        self.side = side

    def area(self):
        print(f" The area of the square is: {self.side **2}")
    
def print_area(shape):
    shape.area()

circle = Circle(5)
square = Square(4)

print_area(circle)
print_area(square)



"""
 The area of the circle is : 78.5
 The area of the square is: 16
 """
    
