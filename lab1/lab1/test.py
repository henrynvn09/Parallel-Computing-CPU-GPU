import subprocess
import random
import sys

# File to modify
filename = 'omp-blocked.cpp'

# List of possible values for TILE_I, TILE_J, TILE_K
possible_values = [8, 256, 512]

# Randomly select a value from the possible_values list for TILE_I, TILE_J, TILE_K
TILE_I = random.choice(possible_values)
TILE_J = random.choice(possible_values)
TILE_K = random.choice(possible_values)

for TILE_I in possible_values:
    for TILE_J in possible_values:
        for TILE_K in possible_values:

            # New lines to replace the first three lines with the selected values
            new_lines = [
                f'#define TILE_I {TILE_I}\n',
                f'#define TILE_J {TILE_J}\n',
                f'#define TILE_K {TILE_K}\n'
            ]

            # Read the existing content of the file
            with open(filename, 'r') as file:
                lines = file.readlines()

            # Replace the first three lines with the new lines
            lines[:3] = new_lines

            # Write the updated content back to the file
            with open(filename, 'w') as file:
                file.writelines(lines)

            # Run the command `make gemm && ./gemm parallel-blocked`
            command = 'make gemm && ./gemm parallel-blocked'

            # Print the selected values
            print(f'TILE_I: {TILE_I}, TILE_J: {TILE_J}, TILE_K: {TILE_K}')
            sys.stdout.flush() 

            # Execute the command
            subprocess.run(command, shell=True)
