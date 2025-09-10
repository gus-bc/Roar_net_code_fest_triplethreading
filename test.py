
villages = [
    [0, 0],
    [16, 25, 464, 2],
    [10, 34, 696, 6],
    [28, 17, 302, 5],
    [19, 57, 523, 10]
]

total_distance = 0
def dist(i, j):
    return abs(villages[i][0] - villages[j][0]) + abs( villages[i][1] - villages[j][1])
def value(i, j):
    global total_distance
    total_distance += dist(i, j)
    val = villages[j][2] - total_distance * villages[j][3]
    return val if val > 0 else 0

total = 0
seq = [0, ]
for i in range(1, len(seq)):
    val = value(seq[i - 1], seq[i])
    print(seq[i - 1], seq[i], val)
    total += val
print(total)