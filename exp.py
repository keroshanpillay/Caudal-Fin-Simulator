import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import concurrent.futures
import time
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import fish

# What are we testing?

TriOp = True
MonoOp = False
ElementOp = False

def calculate_error(params, exp_polynomial):
    
    #default values
    _target = 5
    _amplitude_controller = 1
    
    print(f'Testing params', params)
    freq, amp, shift, element_ratio = params
    try:
        _, _, velocity = fish.swim(freq, amp, shift, _target, _amplitude_controller, element_ratio)
    except Exception as e:
        print(f"An error occurred with fish.swim: {e}")
        return amp, shift, np.inf, freq, element_ratio  # returning a large error to ignore this case
    
    error = (np.abs((velocity-exp_polynomial(freq))/exp_polynomial(freq)))
    return amp, shift, error, freq, element_ratio


def TriOp():
    
    df = pd.read_csv('experimental_data.csv')

    # Extract the frequency and speed data
    tailbeat_frequency = df.iloc[:, 0].values
    speed = df.iloc[:, 1].values

    # Perform a linear regression to get the line of best fit
    coefficients = np.polyfit(tailbeat_frequency, speed, 1)
    exp_polynomial = np.poly1d(coefficients)

    # Define decimal places for rounding
    decimal_places = 5

    # Define parameter ranges
    n = 5
    freq_range = np.linspace(1, 21, n)  # Frequency from 1 to 20 Hz
    amp_range = np.linspace(np.pi/24, np.pi/3, n)  # Flipping amplitude from pi/24 to pi/3
    amp_range = np.round(amp_range, decimals=decimal_places)
    shift_range = np.linspace(0, 2*np.pi, n)  # Phase shift from 0 to 2pi
    shift_range = np.round(shift_range, decimals=decimal_places)
    element_range = np.logspace(-1,1,n)

    # Initialize minimal error and best parameters
    min_error = np.inf
    best_parameters = None
    optimal_freq = {}

    error_list = []
    freq_list = []
    failed_list = []
    print('Testing TriOp')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create all combinations of parameters
        params = list(product(freq_range, amp_range, shift_range, element_range))
        print(f'Testing {len(params)} combinations of parameters')
        # Start the load operations and get future objects
        futures = [executor.submit(calculate_error, param, exp_polynomial) for param in params]

    # Wait for all processes to complete
    concurrent.futures.wait(futures)

    # Gather results and analyze
    for future in futures:
        amp, shift, error, freq, ratio = future.result()
        if error < np.inf:
            if error < min_error:
                min_error = error
                best_parameters = (amp, shift, freq, ratio, error)
            error_list.append((amp, shift, freq, ratio, error))
        else:
            failed_list.append((amp, shift, freq, ratio))

    # Get optimized parameters
    optimal_flipping_amplitude, optimal_phase_shift, optimal_freq, optimal_ratio, error = best_parameters
    # print(f'Optimal Parameters: Amplitude={optimal_flipping_amplitude}, Phase Shift={optimal_phase_shift}, Frequency={optimal_freq[(optimal_flipping_amplitude, optimal_phase_shift)]}Hz')
    optimalParams = {"amp": optimal_flipping_amplitude, "shift": optimal_phase_shift, "freq": optimal_freq, "ratio": optimal_ratio, "error": error}

    # Convert to a pandas DataFrame for easier manipulation
    error_df = pd.DataFrame(error_list, columns=['Amplitude', 'Shift', 'Frequency', 'Element Ratio', 'Error'])
    failed_df = pd.DataFrame(failed_list, columns=['Amplitude', 'Shift', 'Frequency', 'Element Ratio'])
    
    # Save outputs to CSV
    pd.DataFrame([optimalParams]).to_csv('optimal_2.csv', index=False)
    pd.DataFrame(failed_df).to_csv('failed_2.csv', index=False)
    error_df.to_csv('error_df_2.csv', index=False)

    return optimalParams, error_df, freq_list
    
def main():
    start = time.time()
    optimal, error, freq_list = TriOp()
    end = time.time()
    print(f"Optimal Parameters: Amplitude={optimal['amp']}, Phase Shift={optimal['shift']}, Frequency={optimal['freq']}Hz, Element Ratio={optimal['ratio']} with error {optimal['error']}")
    print(f"Time Elapsed: {end-start} seconds")
    # TriPlot(error, freq_list)
    
if __name__ == "__main__":
    main()



