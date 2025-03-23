# Useful when we search names in an unordered list of names
# Runtime is O(n) because if we have:
# list = [1,2,3,4,5,6,7,8]
# if we want to find the number 8, 
# in the worst case it takes 8 searches one by one until we find it
def linear_search(list,target):
    """Returns the index position of the target if found, else returns None
    
    """
    for i in range(0,len(list)):
        if list [i] == target:
            return i
    return None

    
def verify(index):
    if index is not None:
        print("Target found at index:", index)
    else:
        print("Target not found in list")
        
numbers = [1,2,3,4,5,6,7,8,9,19]

result = linear_search(numbers,12)
verify(result)

result = linear_search(numbers,6)
verify(result)