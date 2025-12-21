from day1 import WHEEL_SIZE, INPUT_FILE

EXAMPLE_FILE = "example.txt"

current = 50
num_zeros = 0
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f.readlines():
        num_clicks = int(line[1:])
        num_zeros += num_clicks // WHEEL_SIZE            
        final_rotation = num_clicks % WHEEL_SIZE

        # check for "overflow"
        if line[0] == 'L' and current != 0 and (current - final_rotation) % WHEEL_SIZE > current: # subtraction -> larger num
            num_zeros += 1
        elif line[0] == 'R' and (current + final_rotation) % WHEEL_SIZE < current: # addition -> smaller num
            num_zeros += 1
        elif (current - final_rotation if line[0] == 'L' else current + final_rotation) % WHEEL_SIZE == 0: num_zeros += 1 # neither, end on 0
        current = current - final_rotation if line[0] == 'L' else current + final_rotation
            
        current = current % WHEEL_SIZE
        
print(f'Password is {num_zeros}')