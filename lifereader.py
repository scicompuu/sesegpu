import numpy
import re

def readlife(filename, side):
    """Reads a game of life specification in the Life 1.05 format.
    Only a numpy array is returned, with the specified put in the center of a
    square with the specified side."""
    board = numpy.zeros((side, side), dtype=numpy.bool)
    dx = side // 2
    dy = side // 2

    # Crude, but this will trigger an error if no new point is given
    nowx = -side * 2
    nowy = -side * 2
    with open(filename) as file:
        line = file.readline()
        assert(line.startswith("#Life 1.05"))
        for line in file:
            line = line.strip()
            if line.startswith("#P"):
                match = re.match("#P (-?[0-9]+) (-?[0-9]+)", line)
                nowx = dx + int(match.group(1))
                nowy = dy + int(match.group(2))
            if line.startswith("#"):
                continue
            
            for x, c in enumerate(line):
                board[nowy, nowx + x] = (c == '*')
            nowy += 1
    
    return board



