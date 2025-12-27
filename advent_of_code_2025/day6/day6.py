from typing import List

with open("input.txt", "r", encoding="utf-8") as f:
# with open("example.txt", "r", encoding="utf-8") as f:
    # file = f.readlines()
    # lines = [l.split() for l in file]    
    lines = f.readlines()    
    lines = [l.replace('\n', '') for l in lines]
    

def extract_nums(number_list: List[str]):
    """
        Numbers can also be read left-right.
        e.g.
        64
        23
        314
        +
        = 623 + 431 + 4
        
        64
        2356
        314
        +
        = 623 + 431 + 54 + 6
        
        123
         45
          6
        *
        = 1 * 23 * 356
    """
    rightmost_digit = len(number_list[0])
    extracted_numbers = ['' for i in range(rightmost_digit)]
    
    # Read cols L->R
    for i in range(rightmost_digit):
        for s in number_list:
            # if i < len(s): extracted_numbers[i] += s[i]
            if not s[i] in [' ', '\n']: extracted_numbers[i] += s[i]
    print(extracted_numbers)
            
    return [int(e) for e in extracted_numbers]


def split_line_preserve_spacing(lines: List[str]):
    columns = []
    cur_column = ['' for r in range(num_rows)] # store string-numbers + operation
    for c in range(len(lines[0])):
        if any(lines[r][c] != ' ' for r in range(num_rows)):
            for r in range(num_rows):
                try: cur_column[r] += lines[r][c]
                except IndexError: print(r, c)
        else:
            columns.append(cur_column)
            cur_column = ['' for r in range(num_rows)] # reset
    if cur_column[-1] != '': columns.append(cur_column)
    return columns
    

sum = 0
num_rows, num_cols = len(lines), len(lines[0].split())
columns = split_line_preserve_spacing(lines)
for col in range(num_cols):
    operation = columns[col][-1].replace(' ', '')
    cur_result = 0
    extracted_numbers = extract_nums(columns[col][:-1])
    if operation == '+':
        for num in extracted_numbers: cur_result += num
    else:
        cur_result = extracted_numbers[0]
        for num in extracted_numbers[1:]: cur_result *= num
    sum += cur_result
print(sum)
            


# for col in range(num_cols):
#     operation = lines[-1][col]
#     numbers = [lines[r][col] for r in range(num_rows-1)]
#     cur_result = 0
#     if operation == '+':
#         for row in range(0, num_rows-1): 
#             cur_result += int(lines[row][col])
#     else:
#         cur_result = int(lines[0][col])
#         for row in range(1, num_rows - 1): 
#             cur_result *= int(lines[row][col])
#     sum += cur_result
        
# print(sum)