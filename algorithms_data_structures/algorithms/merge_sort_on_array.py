def merge_sort(list):
    """
    Merge sort algorithm is applied on array (or list in python)
    Sorts a list in ascending order. 
    In this implementation we will return a new list with the sorted values
    Steps:
    Divide: find the midpoint of the list and divide into sublists
    Conquer: recursively sort the sublists created in previous step
    Combine: merge the sorted sublists created in previous step
    
    Takes overall time of the combination of split and merge times so
    that means O(kn log n) time  
    """
    
    if len(list) <= 1:
        return list
    
    left_half, right_half = split(list)
    left = merge_sort(left_half)
    right = merge_sort(right_half)
    
    return merge(left, right)

def split(list):
    """
    Divide the unsorted list at midpoint into sublists
    Returns two sublists - left and right
    
    Takes overall O(log n) time because it divides into half, then half of the half, then half of the half
    
    There is a caveat here, the way we split the list is via slicing, if we check the documentation of python, 
    we will find out that the slicing operation has a runtime of O(k)
    
    That means that overall the split operation takes O(k log n) time
    """
    
    mid = len(list)//2
    left = list[:mid]
    right = list[mid:]
    # we can change this using recursion so the runtim is less
    
    return left, right

def merge(left, right):
    """
    Merges two lists (arrays) sorting them in the process
    Returns a new merged list
    
    Runs overall O(n) times
    """
    
    l = []
    i = 0
    j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            l.append(left[i])
            i += 1
        else:
            l.append(right[j])
            j += 1
            
    while i < len(left):
        l.append(left[i])
        i += 1
        
    while j < len(right):
        l.append(right[j])
        j += 1
        
    return l


def verify_sorted(list):
    """
    python -i algorithms/merge_sort.py 
    False
    True
    """
    n = len(list)
    
    if n == 0 or n == 1:
        return True
    
    return list[0] < list[1] and verify_sorted(list[1:])





alist = [23,22,19,56,55,98,78,33,99]
l = merge_sort(alist)
print(verify_sorted(alist))
print(verify_sorted(l))
            
    