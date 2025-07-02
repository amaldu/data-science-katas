



###----------------------- Exercise 5 ----------------------###

# Create an Employee class with:
# Private attributes for name and access level
# Property to view access level
# A method to promote/demote
# An internal attribute _access_log to keep track of changes


class Employee:
    def __init__(self, name, access_level):
        self.__name = name
        self.__access_level = access_level
        self._access_log = []


    @property 
    def access_level(self):
        return self.__access_level
    
    def promote(self, status):
        if status == "Promotion":
            self.__access_level += 1
            self._access_log.append(f"{self.__name} has been promoted to {self.__access_level} ")
    
    def demote(self, status):
        if self.__access_level > 1 and status == "Demotion":
            self.__access_level -= 1
            self._access_log.append(f"Demoted to level {self.__access_level}")
        else:
            print("Minimum level reached.")

e = Employee("Lucas", 3)
e.promote("Promotion")
e.demote("Demotion")
print(e._Employee__access_level)          
print(e._access_log)  


"""
3
['Lucas has been promoted to 4 ', 'Demoted to level 3']
"""
