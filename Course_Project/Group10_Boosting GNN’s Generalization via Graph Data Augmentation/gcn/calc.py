with open('cora.txt') as f:
    lines = f.readlines()

print(lines[0])

for i in range(2, len(lines), 6):
    nums = [float(x.strip()) for x in lines[i:i+5]]
    print(sum(nums)/5.0)