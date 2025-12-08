INPUT_FILE = "day1_input.txt"
WHEEL_SIZE = 100

def main():
    num_zeros = 0
    current = 50
    with open(INPUT_FILE, "r", encoding='utf-8') as f:
        for line in f.readlines():
            num_clicks = int(line[1:])
            if line[0] == 'L':
                current -= num_clicks
            else:
                current += num_clicks
            if current % WHEEL_SIZE == 0: 
                num_zeros += 1
    print(f"Password is {num_zeros}")

if __name__ == '__main__':
    main()