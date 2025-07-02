



###----------------------- Exercise 5 ----------------------###

# Create an InventoryItem class with:
# Private attributes for item_name and stock and an internal _log (list) attribute to track last update
# Create a function to modify the name of the item and the stock
# methods to add and remove stock
# A read-only property for item name
# 

class InventoryItem:
    def __init__(self, item_name, stock):
        self.__item_name = item_name
        self.__stock = stock
        self._log = []
    
    @property
    def item_name(self):
        return self.__item_name
    
    def add_stock(self, amount):
        self.__stock += amount
        self._log.append(f"Added {amount}")

    def remove_stock(self, amount):
        if amount <= self.__stock:
            self.__stock -= amount
            self._log.append(f"Removed {amount}")
        else:
            print("Not enough stock.")

    @property
    def Stock(self):
        return self.__stock
    
    
item = InventoryItem("Laptop", 10)
item.add_stock(5)
item.remove_stock(3)
print(item.item_name)            
print(item.Stock) 
print(item._log)           


"""
Laptop
12
['Added 5', 'Removed 3']
"""
    
