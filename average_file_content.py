import sys

with open(sys.argv[1], "r") as f:
    line_sum = 0.
    nb_lines = 0
    for line in f:
        line_sum += float(line)
        nb_lines += 1
    line_av = line_sum / nb_lines

with open(sys.argv[1], "w") as f:
    f.write(str(line_av) + '\n')
