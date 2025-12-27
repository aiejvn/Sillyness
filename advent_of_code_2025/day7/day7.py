def part1():
    with open("input.txt", "r", encoding="utf-8") as f:
    # with open("example.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [l.rstrip('\n') for l in lines]
    n_lines = len(lines)

    seen = []

    def rec_get_num_folds(index, line_num):
        # print(line_num, index)
        if line_num >= n_lines:
            return 0
        elif lines[line_num][index] == '^':
            if (line_num, index) not in seen:
                seen.append((line_num, index))
                return 1 + rec_get_num_folds(index-1, line_num+1) + rec_get_num_folds(index+1, line_num+1)
            else:
                return 0
        else: # '.' or 'S'
            return rec_get_num_folds(index, line_num+1)
        
    root = lines[0].find('S')
    print(rec_get_num_folds(root, 1))
    
    
def part2():
    with open("input.txt", "r", encoding="utf-8") as f:
    # with open("example.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [l.rstrip('\n') for l in lines]
    n_lines = len(lines)
    
    ...
   
part2()