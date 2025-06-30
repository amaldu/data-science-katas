



###----------------------- Exercise 3 ----------------------###

# Create a payment gateway system that supports different payment methods: CreditCard, PayPal, Crypto. 
# Each class must implement authenticate() and pay(amount) methods. Write a function checkout(payment_method, amount).

class PaymentMethod:
    def authenticate(self):
        raise NotImplementedError
    
    def pay(self, amount):
        raise NotImplementedError


class CreditCard(PaymentMethod):
    def authenticate(self):
        print(f"Authenticating Credit Card user... ") 
    
    def pay(self, amount):
        print(f"Paid ${amount} using Credit Card.")
    

class PayPal(PaymentMethod):
    def authenticate(self):
        print(f"Authenticating Paypal user... ")  
    
    def pay(self, amount):
        print(f"Paid ${amount} using Paypal.")


class Crypto(PaymentMethod):
    def authenticate(self):
        print(f"Authenticating Crypto user... ")  
    
    def pay(self, amount):
        print(f"Paid ${amount} using Payment.")


def checkout(payment_method: PaymentMethod, amount):
    payment_method.authenticate()
    payment_method.pay(amount)


checkout(PayPal(), 250)
checkout(Crypto(), 0.5)

"""
Authenticating Paypal user... 
Paid $250 using Paypal.
Authenticating Crypto user... 
Paid $0.5 using Payment.
"""
