



###----------------------- Exercise 1 ----------------------###

# Simulate a file system where there are different types of files: TextFile, ImageFile, and AudioFile. 
# Each file should implement a method called open() that returns printed "Opening X File..." and get_metadata() tha returns 
# for TextFile {"type": "text", "encoding": "UTF-8", "lines": 240}
# for ImageFile {"type": "image", "resolution": "1920x1080", "format": "PNG"}
# for AudioFile {"type": "audio", "duration": "3:42", "format": "MP3"}

# Create a function process_files(files: list) that loops through all files, opens them, and prints metadata.


class File:
    def open(self):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError


class TextFile(File):
    def open(self):
        print("Opening text file...")

    def get_metadata(self):
        return {"type": "text", "encoding": "UTF-8", "lines": 240}

class ImageFile(File):
    def open(self):
        print("Opening image file...")

    def get_metadata(self):
        return {"type": "image", "resolution": "1920x1080", "format": "PNG"}

class AudioFile(File):
    def open(self):
        print("Opening audio file...")

    def get_metadata(self):
        return {"type": "audio", "duration": "3:42", "format": "MP3"}
    



def process_files(files):
    for file in files:
        file.open()
        print(file.get_metadata())


files = [TextFile(), AudioFile(), ImageFile()]
process_files(files)

"""
Solution

Opening text file...
{'type': 'text', 'encoding': 'UTF-8', 'lines': 240}
Opening audio file...
{'type': 'audio', 'duration': '3:42', 'format': 'MP3'}
Opening image file...
{'type': 'image', 'resolution': '1920x1080', 'format': 'PNG'}
"""