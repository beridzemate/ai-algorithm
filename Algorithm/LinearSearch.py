def find_number(list, x_number):

    for i in range( len(list) ):
        if list[i] == x_number:
            print(list[i]) 
            return
    print("not found")

numbers = [1,2,3,4,5,6,7,8,9]

find_number(numbers,10)
