import os
import inspect

cwd = os.getcwd()

from .analyses import surrogates

install_loc = str(inspect.getfile(surrogates))[:-22]

os.chdir(install_loc+'hydrogen_delivery_costs')
from .analyses import hydrogen_delivery_costs
os.chdir(cwd)