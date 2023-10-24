#for loop
#list
fruits=['apple', 'orange', 'banana']
for x in fruits:
    print (x)
for x in [0,1,2]:
    print (x)
#range
for x in range(6):
    print (x)

# while loop
i = 1
while i < 6:
    print (i)
    i += 1

#function
def hello(num, name1, name2):
    print ("{}. Hello {}".format(num, name1))
    num += 1
    print(f"{num}. Hello {name2}")
hello(1, "patty","yang")

def five(x):
    print (5 * x)
five(3)




