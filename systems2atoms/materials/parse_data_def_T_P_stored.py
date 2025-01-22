import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

np.set_printoptions(suppress=True)
# Read the data from the text file

def provide_rate_data(data_file):
    data = np.genfromtxt(data_file, delimiter='\t', skip_header=1)

    T_data = data[:, 0]  # Temperature
    pressure_data = data[:, 1]  # Pressure
    r_data = data[:, -2]  # H2_g

    def linear_func(x, a, b):
        return a * x + b

    # Perform curve fitting for each unique pressure value
    unique_pressures = np.unique(pressure_data)

    # Create a dictionary to store fitted parameters for each pressure
    fit_results = {}

    # Iterate through unique pressure values and perform curve fitting
    for pressure in unique_pressures:
        # Select data points for the current pressure
        mask = pressure_data == pressure
        T_data_pressure = T_data[mask]
        r_data_pressure = r_data[mask]

        # Perform curve fitting
        params, covariance = curve_fit(linear_func, 1 / T_data_pressure, np.log(r_data_pressure))

        # Extract the fitted parameters
        a_fit, b_fit = params

        # Store the fit results for this pressure
        fit_results[pressure] = {'a_fit': a_fit, 'b_fit': b_fit}
    return fit_results ##store these for several P and access for new rates

# print(provide_rate_data(sys.argv[1], P=10))