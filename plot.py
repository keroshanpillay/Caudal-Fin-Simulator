import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from matplotlib.ticker import FuncFormatter

def format_ticks(x, pos):
    # this function formats ticks in units of pi
    return f"{round(x/np.pi, 2)}π"

def TriPlot(temp_df, x_var, y_var, z_var, x_label, y_label, z_label, title):
        
    fig = plt.figure(figsize=(10, 8))  # increase plot size
    ax = fig.add_subplot(111, projection='3d')
    
    if x_var == 'Element Ratio':
        temp_df['Element Ratio'] = np.log10(temp_df['Element Ratio'])

    # Original scatter plot, add some transparency
    p = ax.scatter(temp_df[x_var], temp_df[y_var], temp_df[z_var], alpha=0.4)

    # Data for training SVR
    X = temp_df[[x_var, y_var]].values
    y = temp_df[z_var].values
    
    # Standardize the features to improve training of SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the SVR model
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X_scaled, y)

    # Create a grid for plotting the SVR predictions
    xi = np.linspace(temp_df[x_var].min(), temp_df[x_var].max(), 100)
    yi = np.linspace(temp_df[y_var].min(), temp_df[y_var].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Predict with the SVR model
    X_pred = np.vstack([xi.ravel(), yi.ravel()]).T
    X_pred_scaled = scaler.transform(X_pred)
    zi = model.predict(X_pred_scaled).reshape(xi.shape)

    # Plot the SVR prediction surface with a colormap and colorbar
    surf = ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, alpha=1, cmap='viridis', edgecolor='none', linewidth=0.1, zorder=0)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Set up the tick formatter
    # formatter = FuncFormatter(format_ticks)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    
    plt.title(title)  # add a title
    plt.savefig(x_var + '_' + y_var + '_' + z_var + '.png')
    plt.show()
    
def plot_2d(error_df, optimal_freq, optimal_shift, optimal_amp, optimal_ratio):
    fig, axs = plt.subplots(2, 2, figsize=(22, 22))

    # Frequency vs Error
    freq_df = error_df[(error_df['Shift'] == optimal_shift) & (error_df['Amplitude'] == optimal_amp) & (error_df['Element Ratio'] == optimal_ratio)]
    axs[0, 0].plot(freq_df['Frequency'], freq_df['Error'], 'o-')
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_ylabel('Error')
    axs[0, 0].set_title('Frequency vs Error')

    # Phase Shift vs Error
    shift_df = error_df[(error_df['Frequency'] == optimal_freq) & (error_df['Amplitude'] == optimal_amp) & (error_df['Element Ratio'] == optimal_ratio)]
    axs[0, 1].plot(shift_df['Shift'], shift_df['Error'], 'o-')
    axs[0, 1].set_xlabel('Phase Shift')
    axs[0, 1].set_ylabel('Error')
    axs[0, 1].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[0, 1].set_title('Phase Shift vs Error')

    # Amplitude vs Error
    amp_df = error_df[(error_df['Frequency'] == optimal_freq) & (error_df['Shift'] == optimal_shift) & (error_df['Element Ratio'] == optimal_ratio)]
    axs[1, 0].plot(amp_df['Amplitude'], amp_df['Error'], 'o-')
    axs[1, 0].set_xlabel('Flipping Amplitude')
    axs[1, 0].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    axs[1, 0].set_ylabel('Error')
    axs[1, 0].set_title('Amplitude vs Error')
    
    # Element Ratio vs Error
    ratio_df = error_df[(error_df['Frequency'] == optimal_freq) & (error_df['Shift'] == optimal_shift) & (error_df['Amplitude'] == optimal_amp)]
    axs[1, 1].plot(ratio_df['Element Ratio'], ratio_df['Error'], 'o-')
    axs[1, 1].set_xlabel('Element Ratio (1:2)')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_ylabel('Error')
    axs[1, 1].set_title('Element Ratio vs Error')

    plt.tight_layout()
    plt.savefig('2d_plot.png')
    plt.show()


def main():
    # Read data from CSV
    optimal = pd.read_csv('optimal_1.csv').to_dict(orient='records')[0]
    failed_df = pd.read_csv('failed_1.csv') #columns=['Amplitude', 'Shift', 'Frequency', 'Element Ratio']
    error_df = pd.read_csv('error_df_1.csv') #columns=['Amplitude', 'Shift', 'Frequency', 'Element Ratio', 'Error']
    print(f"Optimal params ")
    
    plot_2d(error_df, optimal["freq"], optimal["shift"], optimal["amp"], optimal["ratio"])
    TriPlot(error_df, 'Frequency', 'Amplitude', 'Error', 'Frequency [Hz]', 'Flipping Amplitude [rad]', 'Error', 'Frequency vs Amplitude vs Error')
    TriPlot(failed_df, 'Shift', 'Amplitude', 'Element Ratio', 'Phase Shift [rad]', 'Flipping Amplitude [rad]', 'Element Ratio', 'Phase Shift vs Amplitude vs Element Ratio')
    TriPlot(error_df, 'Element Ratio', 'Shift', 'Error', 'log(Element Ratio)', 'Phase Shift [rad]', 'Error', 'Element Ratio vs Phase Shift vs Error')
    
    
    
'''
We want to plot the following:
1. Frequency, Amplitude vs Error
2. Element Ratio, Shift vs Error

'''
    
if __name__ == '__main__':
    main()