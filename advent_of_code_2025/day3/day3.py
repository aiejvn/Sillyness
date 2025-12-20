with open("input.txt", "r", encoding="utf-8") as f:
# with open("example.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [l.replace("\n", "") for l in lines]
    
voltage = 0


def find_max_in_line(line:str, num_digits:int):
    # Hard repeat process for 12 digits:
    n = len(line)
    max_in_line = 0
    prev = -1
    for digit_index in range(num_digits):
        current = prev+1
        position = num_digits - 1 - digit_index 
        for i in range(current + 1, n-position): 
            if int(line[i]) > int(line[current]): current = i
        # print(prev+1, n-position, current, line[current])
        max_in_line += 10**position * int(line[current])
        prev = current
    return max_in_line
                
for line in lines:
    voltage += find_max_in_line(line,12)

print(voltage)