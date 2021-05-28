from sys import argv

n = int(argv[1])

print(str(n) + ' ' + str(n * (n-1)) + ' directed')
for i in range(n):
    for j in range(n):
        if i != j:
            print(str(i) + ' ' + str(j))
