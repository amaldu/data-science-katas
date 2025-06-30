



###----------------------- Exercise 6 ----------------------###

# Write two classes, TextFileWriter and JSONFileWriter, that both implement a method write(data) but behave differently. 
# Then write a function save_file(writer, data) that uses duck typing to call write() on any writer passed.




class TextFileWriter:
    def write(self, data):
        print("Writing text data:", data)


class JSONFileWriter:
    def write(self, data):
        print("Writing JSON data:", {"content": data})
    
def save_file(writer, data):
    writer.write(data)


text_writer = TextFileWriter()
json_writer = JSONFileWriter()

save_file(text_writer, "Hello, world!")
save_file(json_writer, "Hello, world!")


"""
Writing text data: Hello, world!
Writing JSON data: {'content': 'Hello, world!'}
"""