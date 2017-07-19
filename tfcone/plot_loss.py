import matplotlib.pyplot as plt
import numpy as np
import re

infile = "nohup.out"

losses = []

with open(infile) as f:
    f = f.readlines()

    pattern = re.compile("Loss: (\d+)")

    for line in f:
        for match in re.finditer( pattern, line ):
            losses.append( int( match.groups()[0] ) )

plt.plot( losses )
plt.show()


