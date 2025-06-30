



###----------------------- Exercise 1 ----------------------###

# Simulate a file system where there are different types of files: TextFile, ImageFile, and AudioFile. 
# Each file should implement a method called open() that returns printed "Opening X File..." and get_metadata() tha returns 
# for TextFile {"type": "text", "encoding": "UTF-8", "lines": 240}
# for ImageFile {"type": "image", "resolution": "1920x1080", "format": "PNG"}
# for AudioFile {"type": "audio", "duration": "3:42", "format": "MP3"}

# Create a function process_files(files: list) that loops through all files, opens them, and prints metadata.










































"""
Solution

Opening text file...
{'type': 'text', 'encoding': 'UTF-8', 'lines': 240}
Opening audio file...
{'type': 'audio', 'duration': '3:42', 'format': 'MP3'}
Opening image file...
{'type': 'image', 'resolution': '1920x1080', 'format': 'PNG'}
"""