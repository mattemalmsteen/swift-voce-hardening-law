import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from datetime import datetime

# Get the path of the current file
path = os.path.dirname(os.path.abspath(__file__))

# Introduction messages
print("This script calculates the Swift-Voce approximation starting from experimental engineering data.")
print("To run the script, a .txt file with tensile test engineering data is needed: elongation in % and load in MPa.")
print("The file and the script must be in the same folder, and the data in the file must be placed in two columns (elongation and load) without any symbols.")
print("The script will create: True Curve, Plastic True Curve, Swift Approx., Voce Approx., Swift-Voce approx. at 100% deformation, and the parameters for each approximation.")

input("Press Enter to start.")
file_name = input("Insert the name of the raw data file without extension: ")
material_name = input("Enter material name: ")
supplier_name = input("Enter supplier name: ")

today = datetime.now()
timestamp = today.strftime("%Y_%m_%d_%H_%M")
data_file = os.path.join(path, f"{file_name}.txt")

try:
    data = np.genfromtxt(data_file)
except IOError:
    print(f"Error: File {data_file} not found.")
    exit()

# Extracting elongation and load
elongation_percent = data[:, 0]
load = data[:, 1]
elongation = elongation_percent / 100

# Plotting the imported curve
plot_eng = input("Want to plot the imported curve? (y/n) ").strip().lower()
if plot_eng == "y":
    plt.plot(elongation_percent, load)
    plt.title("Experimental Curve")
    plt.xlabel("Elongation (%)")
    plt.ylabel("Load (MPa)")
    plt.show()

# Determine necking position
neck_pos = 1 + load.argmax()
neck_load = load[:neck_pos]
neck_elongation = elongation[:neck_pos]

# Convert to true stress and true strain
true_elongation = np.log(neck_elongation + 1)
true_load = neck_load * (1 + neck_elongation)

np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_TrueCurve.txt"), np.column_stack([true_elongation, true_load]))

# Plot the True Curve
plot_tc = input("Want to plot the True Curve? (y/n) ").strip().lower()
if plot_tc == "y":
    plt.plot(true_elongation, true_load)
    plt.title("True Curve")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.show()

# Input yield strength
yield_strength = float(input("Enter Yield strength value: "))

# Function to find nearest value in TrueS to Yield
def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]

yield_pos = 1 + np.abs(true_load - yield_strength).argmin()
plastic_load = true_load[yield_pos:]
plastic_elongation = true_elongation[yield_pos:] - true_elongation[yield_pos]

np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_TruePlasticCurve.txt"), np.column_stack([plastic_elongation, plastic_load]))

# Plot the True Plastic Curve
plot_tpc = input("Want to plot the Plastic True Curve? (y/n) ").strip().lower()
if plot_tpc == "y":
    plt.plot(plastic_elongation, plastic_load)
    plt.title("True Plastic Curve")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.show()

# Voce Approximation
def voce_error(params):
    Q, B = params
    voce_load = yield_strength + Q * (1 - np.exp(-B * plastic_elongation))
    return np.sum((plastic_load - voce_load) ** 2)

def voce_constraint(params):
    Q, B = params
    return Q - ((Q * (B + 1) * np.exp(-B * np.max(plastic_elongation))) - yield_strength)

initial_guess = (1, 1)
constraints = {'type': 'eq', 'fun': voce_constraint}
result_voce = minimize(voce_error, initial_guess, method='SLSQP', constraints=constraints)
Q_voce, B_voce = result_voce.x
voce_load = yield_strength + Q_voce * (1 - np.exp(-B_voce * plastic_elongation))

np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_Voce.txt"), np.column_stack([plastic_elongation, voce_load]))
np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_Voce_QeBeta_Param.txt"), np.array([[Q_voce, B_voce]]))

# Swift Approximation
def swift_error(params):
    ei = params
    swift_load = (yield_strength / (ei ** (ei + np.max(plastic_elongation)))) * ((plastic_elongation + ei) ** (ei + np.max(plastic_elongation)))
    return np.sum((plastic_load - swift_load) ** 2)

initial_guess_swift = (0.00001,)
result_swift = minimize(swift_error, initial_guess_swift, method='Nelder-Mead')
ei_swift = result_swift.x[0]
swift_load = (yield_strength / (ei_swift ** (ei_swift + np.max(plastic_elongation)))) * ((plastic_elongation + ei_swift) ** (ei_swift + np.max(plastic_elongation)))

np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_Swift.txt"), np.column_stack([plastic_elongation, swift_load]))
np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_Swift_e0_Param.txt"), np.array([[ei_swift]]))

# Swift-Voce Approximation
while True:
    alpha = float(input("Enter a value for alpha to calculate the S-V approx.: "))
    ep1 = np.arange(0, 0.08, 0.002)
    ep2 = np.arange(0.08, 1.02, 0.02)
    ep100 = np.concatenate([ep1, ep2])
    sv_load = (1 - alpha) * (yield_strength + Q_voce * (1 - np.exp(-B_voce * ep100))) + (alpha * (yield_strength / (ei_swift ** (ei_swift + np.max(plastic_elongation)))) * ((ep100 + ei_swift) ** (ei_swift + np.max(plastic_elongation))))

    plt.plot(plastic_elongation, plastic_load, label='Experimental Data')
    plt.plot(ep100, sv_load, label='Swift-Voce Approximation')
    plt.title("Swift-Voce Approximation")
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.legend(loc='lower right')
    plt.show()

    recalculate = input("Want to recalculate with another value of alpha? (y/n) ").strip().lower()
    if recalculate == "n":
        break

np.savetxt(os.path.join(path, f"{timestamp}_{material_name}_{supplier_name}_SwiftVoce_def100_Alpha_{alpha}.txt"), np.column_stack([ep100, sv_load]))
