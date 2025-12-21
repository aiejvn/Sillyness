from typing import List

with open("input.txt", "r", encoding="utf-8") as f:
# with open("example.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [list(l.replace("\n", "")) for l in lines]
    

def num_rolls_up_and_down(up_valid: bool, down_valid:bool, i:int, j:int):
    """Check above and below a certain square."""
    num_rolls = 0
    if up_valid and lines[i-1][j] == '@': num_rolls += 1
    if down_valid and lines[i+1][j] == '@': num_rolls += 1
    return num_rolls

def get_accessible_rolls(lines:List[str]):
    num_accessible = 0
    accessible_coords = []
    h,l = len(lines), len(lines[0])
    for i in range(h):
        for j in range(l):
            if lines[i][j] == '@':
                left_valid = j-1>=0
                right_valid = j+1<l
                up_valid = i-1>=0
                down_valid = i+1<h
                
                left=0
                if left_valid:
                    left = num_rolls_up_and_down(up_valid, down_valid, i, j-1)
                    if lines[i][j-1] == '@': left += 1
                    
                center = num_rolls_up_and_down(up_valid, down_valid, i, j)
                
                right=0
                if right_valid:
                    right = num_rolls_up_and_down(up_valid, down_valid, i, j+1) 
                    if lines[i][j+1] == '@': right += 1
            
                if left+center+right < 4: 
                    num_accessible += 1
                    accessible_coords.append((i,j))
                    
    return num_accessible, accessible_coords

num_removed = 0
while True:
    num_accessible, coords = get_accessible_rolls(lines)
    if num_accessible == 0: break
    
    num_removed += num_accessible
    for x,y in coords:
        lines[x][y] = '.'
        
print(num_removed)