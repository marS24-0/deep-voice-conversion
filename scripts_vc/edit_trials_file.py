import sys
if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} input_path output_path')
    exit(1)
with open(sys.argv[1], 'r') as infile:
    a = [x.strip().split() for x in infile.readlines()]
b = [[x[0], f'{x[1]}_{x[0]}', x[2]] for x in a]
c = [' '.join(x) for x in b]
with open(sys.argv[2], 'w') as outfile:
    outfile.write('\n'.join(c))

