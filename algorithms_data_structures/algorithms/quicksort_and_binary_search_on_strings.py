
# This approach uses binary search that is way faster than linear search BUT it needs the names to be sorted before!
# The runtime of binary search is O(log n), it's faster than linear search



def quick_sort(values):
    if len(values) <= 1:
        return values
    less_than_pivot = []
    grater_than_pivot = []
    pivot = values[0]
    for value in values[1:]:
        if value <= pivot:
            less_than_pivot.append(value)
        else:
            grater_than_pivot.append(value)
    print("%15s %1s %-15s" % (less_than_pivot, pivot, grater_than_pivot))
    return quick_sort(less_than_pivot) + [pivot] + quick_sort(grater_than_pivot)

names = ["Ana", "Carlos", "Elena", "Javier", "María", "Roberto", "Sofía", "Diego", "Lucía", "Fernando"]

sorted_names = quick_sort(names)
for name in sorted_names:
    print(name)
    
#     [] Ana ['Carlos', 'Elena', 'Javier', 'María', 'Roberto', 'Sofía', 'Diego', 'Lucía', 'Fernando']
#              [] Carlos ['Elena', 'Javier', 'María', 'Roberto', 'Sofía', 'Diego', 'Lucía', 'Fernando']
#       ['Diego'] Elena ['Javier', 'María', 'Roberto', 'Sofía', 'Lucía', 'Fernando']
#    ['Fernando'] Javier ['María', 'Roberto', 'Sofía', 'Lucía']
#       ['Lucía'] María ['Roberto', 'Sofía']
#              [] Roberto ['Sofía']      
# Ana
# Carlos
# Diego
# Elena
# Fernando
# Javier
# Lucía
# María
# Roberto
# Sofía

def binary_search(collection,target):
    """first and last variables refer to value of the indexes of the list
    """
    first = 0
    last = len(collection) - 1
    
    while first <= last:
        midpoint = (first + last) //2 # "//" rounds to the unit
        
        if collection[midpoint] == target:
            return midpoint
        elif collection[midpoint] < target:
            first = midpoint + 1
        else:
            last = midpoint - 1
            
    return None


for n in sorted_names:
    index = binary_search(sorted_names,n)
    print(index)
    
for n in sorted_names:
    index = binary_search(sorted_names,"Sofía")
    print(index)
