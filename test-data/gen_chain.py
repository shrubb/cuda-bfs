from sys import argv

n = int(argv[1])

with open('chain' + str(n) + '.txt', 'w') as f:
    f.write(str(n) + ' ' + str(n-1) + ' directed\n')
    for i in range(n):
        f.write(str(i) + ' ' + str(i+1) + '\n')

