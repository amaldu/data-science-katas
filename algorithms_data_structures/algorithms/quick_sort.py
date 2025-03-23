


# This sorting algorithm is recursive. 
# 1. It divides the list into 2 sublists (left and right) keeping the value used to divide them, isolated. 
# This value is called pivot
# The behaviour in the next steps happens symmetrically on both branches with the sublists but 
# we only describe what happens to the left sublist.
# 2. On the left sublist it applies the same algorithm that keeps the central value and divides the sublist
# again into two subsublists (left and right). 
# 3. It sorts the subsublists and sends them back to create the sublist.
# 4. The sublists are ordered and sent back to the list.

# Runtime is O(n²)
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

numbers = [5,67,38,33,12,88,6,90,32]

sorted_numbers = quick_sort(numbers)
print(sorted_numbers)