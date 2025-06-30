



###----------------------- Exercise 8 ----------------------###

# You have multiple ways to send notifications: EmailSender, SMSSender, PushSender. 
# Each must implement a method send(message). Write a function notify(sender, msg) that works for any sender using duck typing.

class EmailSender:
    def send(self, message):
        print(f"Email sent")
    
class SMSSender:
    def send(self, message):
        print(f"SMS sent")
    
class PushSender:
    def send(self, message):
        print(f"Push message sent")

def notify(sender,message):
    sender.send(message)


email_sender = EmailSender()
sms_sender = SMSSender()
push_sender = PushSender()

notify(email_sender, "Dear guests, \n please confirm the assistance to the party \n Best ")
notify(sms_sender, "I dnt knw wat time?")
notify(push_sender, "Please confirm the code")


"""
Email sent
SMS sent
Push message sent
 """



