



###----------------------- Exercise 9 ----------------------###


# Create two logger classes: ConsoleLogger and FileLogger. 
# Both should have a method log(message). 
# Write a function write_log(logger, message) that accepts any logger using duck typing.

class ConsoleLogger:
    def log(self, message):
        print(f"Logging in the console...")


class FileLogger:
    def log(self, message):
        print(f"Logging in the file...")

def write_log(logger, message):
    logger.log(message)

log_console = ConsoleLogger()
log_file = FileLogger()

write_log(log_console, "log.info('updating file')")
write_log(log_file, "log.info('table updated')")



"""
Logging in the console...
Logging in the file...
"""

