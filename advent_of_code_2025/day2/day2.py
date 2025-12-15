import math

ranges = []

# with open("./example.txt", "r", encoding="utf-8") as f:
with open("./day2_input.txt", "r", encoding="utf-8") as f:
    ranges_as_string = f.read().split(",")
    for elem in ranges_as_string:
        hyphen_index = elem.find("-")
        lower, upper = int(elem[:hyphen_index]), int(elem[hyphen_index+1:])
        ranges.append((lower, upper))

def get_relevant_factors(n):
    relevant_factors = []
    for i in range(1, n):
        if n % i == 0:
            relevant_factors.append(i)
    return relevant_factors 
    
sum = 0
# Find invalid numbers
for start, end in ranges:
    for num in range(start, end+1):
        num_as_str = str(num)
        n = len(num_as_str)
        factors = get_relevant_factors(n)
        for factor in factors:
            works = True
            for i in range(n // factor - 1):
                if num_as_str[i*factor:(i+1)*factor] != num_as_str[(i+1)*factor:(i+2)*factor]:
                    works = False
                    break
                
            if works: 
                sum += num
                break
        
        
print(sum)