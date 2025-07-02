



###----------------------- Exercise 2 ----------------------###


# Create a BankAccount class with a private balance, an internal attribute _transaction_count that tracks the number of operations and owner
# Create methods to deposit and withdraw, check_balance (with a getter) to check the current balance. 
# Prevent negative deposits and withdrawals where the amount doesn't exceed the balance

class BankAccount:
    def __init__(self, owner,  balance):
        self.owner = owner
        self.__balance = balance
        self._transaction_count = 0

    def deposit(self,amount):
        if amount > 0:
            print(f"The amount of {amount} has been deposited...")
            self.__balance += amount
            self._transaction_count += 1
        else:
            print("Incorrect amount.")

    def withdraw(self,amount):
        if self.__balance >= amount > 0:
            print(f"The amount of {amount} has been withdrawn...")
            self.__balance -= amount
            self._transaction_count += 1
        else:
            print("Insufficient funds.")

    @property
    def check_balance(self):
        return self.__balance
    
acc = BankAccount("Alice", 1000)
acc.deposit(200)
acc.withdraw(150)
print(acc.check_balance)             
print(acc._transaction_count)

"""The amount of 200 has been deposited...
The amount of 150 has been withdrawn...
1050
2"""

