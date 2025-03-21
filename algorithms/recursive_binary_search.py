def recursive_binary_search(list, target):
    """ 
    this version wont return the value, instead True or False
    
    this a recursive function, a function that calls itself. 
    
    the aount of times a function calls itself is called recursion depth. 
    python has recursion max depth, it prefers iteration
    
    iterative solution = loop structure of some kind
    """
    if len(list) == 0:
        return False
    else:
        midpoint = (len(list))//2
        
        if list[midpoint] == target:
            return True
        else:
            if list[midpoint] < target:
                return recursive_binary_search(list[midpoint + 1:], target) # midpoint all way to the end
            else: 
                return recursive_binary_search(list[:midpoint], target)

def verify(result):
    print("Target found:", result)
    
numbers = [1,2,3,4,5,6,7,8,9,10]

result = recursive_binary_search(numbers,12)
verify(result)

result = recursive_binary_search(numbers,6)
verify(result)