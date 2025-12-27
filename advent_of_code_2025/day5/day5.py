"""
O(n^2) check for every number, for every range, if number is in range.
Expand out ranges to lists of numbers to avoid O(n^3) checking 
"""
import re

range_regex = re.compile(r"(\d+)-(\d+)")

ranges = []
with open("input.txt", "r", encoding="utf-8") as f:
# with open("example.txt", "r", encoding="utf-8") as f:
    curline = f.readline()
    while True: 
        match = range_regex.search(curline)
        if not match: break
        start = int(match.group(1))
        end = int(match.group(2))
        # ranges.append(range(start, end+1))
        ranges.append((start, end))
        curline = f.readline()

# Merge ranges
ranges = sorted(ranges, key = lambda x: x[0])
total_ranges = [ranges[0]]
for cur_start, cur_end in ranges[1:]:
    prev_start, prev_end = total_ranges[-1]
    
    # prev_start < cur_start is always true
    if prev_end < cur_end:
        if prev_end < cur_start:
            total_ranges.append((cur_start, cur_end))
        else:
            total_ranges[-1] = (prev_start, cur_end) # extend range
        
num_fresh = 0
for (start, end) in total_ranges:
    num_fresh += end - start + 1
    # print(start, end)
print(num_fresh)
        
#     numbers = f.readlines()


# num_fresh = 0
# for num in numbers:
#     for interval in ranges:
#         if int(num) in interval: 
#             num_fresh += 1
#             break

# print("Num Fresh:", num_fresh)
# print("Num Rotten:", len(numbers) - num_fresh)