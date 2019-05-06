__index = []

def quick_sort(left, right):
    if left > right:
        return
    temp = __index[left][1]
    i = left
    j = right
    while i != j:
        while __index[j][1] <= temp and i < j:
            j = j - 1
        while __index[i][1] >= temp and i < j:
            i = i + 1
        if i < j:
            t = __index[i]
            __index[i] = __index[j]
            __index[j] = t
    
    t = __index[left]
    __index[left] = __index[i]
    __index[i] = t
        
    quick_sort(left, i - 1)
    quick_sort(i + 1, right)

 
def quicksort(index, left, right):
    global __index
    __index = index
    quick_sort(left, right)
    return __index
#    for i in range(100):
#        print(str(i) + " " + str(__index[i][1]))

