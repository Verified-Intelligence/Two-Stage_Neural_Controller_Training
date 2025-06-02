with open ('output_turbo_tll.log', 'r') as file:
    for line in file:
        if line.startswith("Result"):
            print(line.strip())