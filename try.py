# 加減乘除練習
print(10 + 5)
print(10 - 5)
print(10 * 5)
print(10 / 5)
print(10 % 5)
print(10 % 3)
print(10 ** 2)

#functions
print(abs(5))
print(abs(-5))
print(pow(2,10))
print(max(4, 2, 1, -10, 100))
print(min(4, 2, 1, -10, 100))
# x.5, the value will be rounded up if the rounded up value is an even number.
# otherwise, it will be rounded down.
print(round(2.5)) 
print(round(3.5))
print(int(3.0))
print(float(3))
print(str(3) + str(3))
print(3 + 3)

#math functions
import math
print(math.sqrt(9))
print(math.pi)
print(math.e)
print(math.floor(4.99))
print(math.ceil(4.01))


x = 5
x = x + 1
print(x)
x = 5
x += 1
print(x)

#string
print("hello")
print("hello"[0])

mystring = "hello"
print(mystring[0])
print(mystring[-4])
print(mystring[-3])
print(mystring[-2])
print(mystring[-1])
print(mystring[-5])

# string slicing [start:end:stepsize]
x = "abcdefg"
print(x[2:])
print(x[2:5])
print(x[0:6:2])
print(x[::-1])
print(x[::2])

print('I said "goodbye".')
print("I said 'goodbye'")
print("\n\"Goodbye \"")

str1 = 'hello'
str2 = 'goodbye'
print(str1 + " " + str2 + "\n" + str(3))

#string is immutable
print(str1[0])
# str1[0]='H'
print(str1)


str = "Aloha"
print(len(str))
print(str.upper())
print(str)
print(str.lower())
str=str.upper()
print(str)

strlow='lower'
strup='UPPER'
print(strlow.islower())
print(strup.islower())
print(strlow.isupper())
print(strup.isupper())

name="patty"
print(name.index("a"))

#replace 
name = "patty kuo"
print(name.replace("p","P"))
print(name.replace("patty","sharon"))

#split and list
sentence = "today is  a good day"
print(sentence.split(" "))  
print(list(sentence))

#format
print("I have a string {}".format("here it is"))
print("I have a string {} {} {}".format(1,"2",3))
print("{name},{address},{age}".format(name="patty", address="taiwan", age=18))

#fstring
name = "patty"
age = 18
print(f"hello i am {name}, I am {age} year old")