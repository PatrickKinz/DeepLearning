print('Hello World')

string = 'Hydrogen'

# %% codecell

for index, letter in enumerate(string):
    print((letter,index))

# %% codecell

import numpy as np
import matplotlib.pyplot as plt

# %% codecell
x = np.linspace(0,10,500)
plt.plot(x, np.sin(x))

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

plt.plot(x, np.sin(x))
import matplotlib as mpl
mpl.style.use('ggplot')
plt.plot(x, np.sin(x))
