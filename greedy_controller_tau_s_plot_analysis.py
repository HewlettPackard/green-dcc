
#%%
import matplotlib.pyplot as plt
import numpy as np

# Data for Carbon Intensity Greedy
tau_s = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
ci_energy = [788.31, 779.24, 770.43, 775.79, 786.98, 809.85, 848.97, 912.40, 1017.20]
ci_energy_std = [171.62, 159.76, 147.70, 135.95, 126.21, 115.57, 107.09, 110.90, 148.98]
ci_carbon = [249873.07, 246279.39, 242862.83, 243371.29, 245580.71, 251001.15, 260893.11, 277474.90, 305614.54]
ci_carbon_std = [59086.07, 55981.63, 52892.17, 50261.56, 48249.04, 46144.78, 44839.17, 46614.44, 57481.83]
ci_water = [1417.15, 1417.79, 1415.70, 1423.07, 1430.15, 1441.30, 1457.28, 1478.40, 1505.62]
ci_water_std = [211.76, 211.53, 211.24, 209.72, 207.82, 205.35, 202.94, 201.10, 201.59]

# Data for Temperature Greedy
temp_energy = [783.76, 777.11, 769.90, 775.05, 783.83, 802.50, 835.75, 891.32, 984.24]
temp_energy_std = [167.69, 157.55, 147.05, 135.16, 124.45, 111.71, 100.26, 97.43, 124.59]
temp_carbon = [249938.63, 247277.49, 244467.02, 245124.73, 246719.65, 251053.52, 259510.60, 274245.26, 299586.21]
temp_carbon_std = [57341.67, 54580.51, 51741.14, 48979.79, 46519.31, 43744.39, 41776.67, 42312.76, 50499.13]
temp_water = [1415.67, 1417.23, 1415.95, 1424.05, 1431.17, 1441.98, 1457.56, 1478.43, 1505.55]
temp_water_std = [210.45, 209.67, 208.60, 207.18, 205.37, 203.37, 201.68, 200.58, 201.66]

# Plot settings
plt.figure(figsize=(4.5, 8))

# Subplot for Energy
plt.subplot(3, 1, 1)
plt.errorbar(tau_s, ci_energy, yerr=np.array(ci_energy_std)/10, label="CI Greedy", marker='o', linestyle='-', capsize=4)
plt.errorbar(tau_s, temp_energy, yerr=np.array(temp_energy_std)/10, label="Temp Greedy", marker='s', linestyle='--', capsize=4)
plt.xlabel(r"$\tau_s$ (Load Saturation Threshold)")
plt.ylabel("Energy (kWh)")
plt.title("Energy Consumption")
plt.legend()
plt.grid(True)

# Subplot for Carbon Emissions
plt.subplot(3, 1, 2)
plt.errorbar(tau_s, np.array(ci_carbon)/1000, yerr=np.array(ci_carbon_std)/10000, label="CI Greedy", marker='o', linestyle='-', capsize=4)
plt.errorbar(tau_s, np.array(temp_carbon)/1000, yerr=np.array(temp_carbon_std)/10000, label="Temp Greedy", marker='s', linestyle='--', capsize=4)
plt.xlabel(r"$\tau_s$ (Load Saturation Threshold)")
plt.ylabel("Carbon Emissions (Mg CO$_2$)")
plt.title("Carbon Emissions")
plt.legend()
plt.grid(True)

# Subplot for Water Usage
plt.subplot(3, 1, 3)
plt.errorbar(tau_s, ci_water, yerr=np.array(ci_water_std)/10, label="CI Greedy", marker='o', linestyle='-', capsize=4)
plt.errorbar(tau_s, temp_water, yerr=np.array(temp_water_std)/10, label="Temp Greedy", marker='s', linestyle='--', capsize=4)
plt.xlabel(r"$\tau_s$ (Load Saturation Threshold)")
plt.ylabel("Water Usage (liters)")
plt.title("Water Usage")
plt.legend()
plt.grid(True)

# Adjust layout and save
plt.tight_layout()

plt.savefig('Figures/tau_s_CI_vs_temp_greedy.pdf', format='pdf')

plt.show()
#%%