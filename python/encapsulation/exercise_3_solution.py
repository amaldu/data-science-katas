


###----------------------- Exercise 3 ----------------------###

# Create a Student class that:
# Has private attributes for name and grades (list)
# Has a @property to access name
# Allows modifying name with a setter
# Has an internal _id to simulate an internal database reference "ID_twofirstcapitallettersofthesame123"
# Has an attribute called average_grade that returns the average of the grades of the person. 

class Student:
    def __init__(self, name, grades):
        self.__name = name
        self.__grades = grades
        self._id = f"ID_{name[:2].upper()}123"

    @property 
    def Name(self):
        return self.__name
    
    @Name.setter
    def Name(self, value):
        self.__name = value

    def average_grade(self):
        return sum(self.__grades)/len(self.__grades)
    
s = Student("Carlos", [90, 85, 92])
print(s.Name)                 
print(s.average_grade())     
print(s._id)   
    
"""
Carlos
89.0
ID_CA123
"""

