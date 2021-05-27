from sys import argv

n = int(argv[1])

with open('full' + str(n) + '.txt', 'w') as f:
    f.write(str(n) + ' ' + str(n * (n-1)) + ' directed\n')
    for i in range(n):
        for j in range(n):
            if i != j:
                f.write(str(i) + ' ' + str(j) + '\n')

