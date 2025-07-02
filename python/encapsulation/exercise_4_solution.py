


###----------------------- Exercise 4 ----------------------###

# Create a Car class that:
# Has private _speed and a public interface to set it
# Disallows setting speed below 0 or above 300, otherwise print a statement giving instructions
# Exposes current speed via a property
# Uses _engine_temp as an internal-only attribute = 70 degrees

class Car:
    def __init__(self, speed):
        self.__speed = speed
        self._engine_temp = 70

    @property
    def speed(self):
        return self.__speed
    
    @speed.setter
    def speed(self, speed_value):
        if 0 < speed_value <= 300:
            self.__speed = speed_value

        else:
            print("Speed must be between 0 and 300")

c = Car("Tesla Model Y")
c.speed = 250
print(f"The current speed if {c.speed}")               
print(f"The temperature of the engine is {c._engine_temp}") 


"""
The current speed if 250
The temperature of the engine is 70
"""