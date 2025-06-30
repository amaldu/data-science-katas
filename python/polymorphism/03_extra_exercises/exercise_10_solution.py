




###----------------------- Exercise 10 ----------------------###

# You have two classes: CreditCardPayment and PayPalPayment, both with a method pay(amount). 
# Write a function process_payment(payment_method, amount) that uses duck typing to call the correct method.

class CreditCardPayment:
    def pay(self, amount):
        print(f"payment of {amount} via credit done")

class PayPalPayment:
    def pay(self, amount):
        print(f"payment of {amount} via PayPal done")

def process_payment(payment_method, amount):
    payment_method.pay(amount)

process_payment(CreditCardPayment(), 60000)
process_payment(PayPalPayment(), 1000)


"""
payment of 60000 via credit done
payment of 1000 via PayPal done
"""
