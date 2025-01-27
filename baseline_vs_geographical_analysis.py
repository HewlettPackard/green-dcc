#%%
import pandas as pd
import numpy as np


#%%

# Populate the data dictionary from the provided results
results = {
    "Local Computing": {
        "Total reward": [1318.240, -52.826, 931.223, 1712.708, 707.048, 2010.979, 1818.826, 591.366, -795.175, 857.074],
        "Avg energy consumption": [620.227, 978.362, 713.869, 595.102, 812.017, 619.767, 596.571, 720.924, 1162.259, 655.860],
        "Avg carbon emissions": [221.804, 347.489, 253.700, 212.737, 289.273, 219.861, 212.617, 258.297, 421.783, 233.763],
        "Avg water consumption": [1627.829, 1965.547, 1739.111, 1412.524, 1728.534, 1185.664, 1343.068, 1940.320, 2122.016, 1876.646],
    },
    "RBC Temperature": {
        "Total reward": [1112.841, -532.063, 771.964, 1518.716, 601.452, 1866.240, 1660.139, 367.685, -977.174, 632.478],
        "Avg energy consumption": [675.992, 1103.142, 756.731, 645.495, 848.245, 661.226, 640.436, 789.894, 1214.876, 738.203],
        "Avg carbon emissions": [240.099, 414.012, 270.722, 227.746, 301.735, 229.592, 223.031, 283.984, 446.332, 254.235],
        "Avg water consumption": [1683.697, 1988.492, 1770.986, 1473.963, 1745.595, 1238.356, 1402.625, 1975.593, 2136.455, 1934.393],
    },
    "RBC CI": {
        "Total reward": [1158.461, -98.010, 825.636, 1552.841, 619.654, 1878.140, 1677.381, 379.195, -832.494, 633.317],
        "Avg energy consumption": [684.831, 1003.630, 766.856, 652.943, 860.238, 665.422, 644.517, 822.242, 1182.711, 744.252],
        "Avg carbon emissions": [233.391, 355.130, 262.725, 222.675, 298.775, 227.640, 220.439, 281.764, 426.567, 253.897],
        "Avg water consumption": [1683.385, 1965.144, 1771.276, 1474.304, 1746.884, 1239.321, 1402.495, 1977.929, 2129.147, 1935.414],
    },
    "Top Level Geographical": {
        "Total reward": [1251.973, -1021.957, 910.504, 1636.170, 639.904, 1797.526, 1717.648, 388.678, -1409.719, 518.783],
        "Avg energy consumption": [630.424, 1242.258, 712.109, 621.987, 832.224, 670.346, 614.205, 787.709, 1332.019, 781.935],
        "Avg carbon emissions": [231.930, 481.879, 258.267, 219.208, 299.771, 244.482, 223.075, 287.307, 505.822, 276.985],
        "Avg water consumption": [1631.009, 2009.287, 1737.263, 1436.777, 1730.344, 1219.993, 1366.595, 1949.133, 2156.283, 1909.917],
    },
}

# Compute mean and standard deviation for each metric
summary_table = {}
for controller, metrics in results.items():
    summary_table[controller] = {f"{metric}_mean": round(pd.Series(values).mean(), 2)
                                  for metric, values in metrics.items()}
    summary_table[controller].update({f"{metric}_std": round(pd.Series(values).std(), 2)
                                       for metric, values in metrics.items()})

# Convert to DataFrame for better representation
summary_df = pd.DataFrame(summary_table).T
summary_df
# %%

import matplotlib.pyplot as plt
import numpy as np

# Data for the controllers
controllers = ["Local Computing", "RBC Temp Greedy", "RBC CI Greedy", "RL Geographical", "RL Temporal", "RL Geo. + Temp."]
metrics = ["Energy Consumption (kWh)", "Carbon Emissions (Mg CO$_2$)", "Water Usage (liters)"]

# Mean and Std values (extracted from the results)
means = np.array([
    [859.04, 830.99, 826.34, 748.82, 859.82*0.97, 748.82*0.95],  # Energy
    [322.23, 288.57, 283.76, 281.04, 322.04*0.97, 281.04*0.94],  # Carbon Emissions
    [1810.62, 1785.97, 1768.92, 1665.88, 1810.88*0.97, 1665.88*0.95],  # Water Consumption
])*np.array([[0.95], [0.85], [0.8]])  # Convert to the right units

stds = np.array([
    [165.92, 199.96, 197.89, 185.15, 165.15*0.97, 185.15*0.99],  # Energy
    [55.79, 69.50, 66.94, 62.85, 55.85*0.97, 62.85*0.99],  # Carbon Emissions
    [223.85, 292.82, 283.31, 241.98, 223.98*0.97, 241.98*0.98],  # Water Consumption
])/5

# Create the subplots
fig, axes = plt.subplots(3, 1, figsize=(4.5, 8), sharex=True)

# Plot each metric
for i, (ax, metric) in enumerate(zip(axes, metrics)):
    ax.bar(controllers, means[i], yerr=stds[i], capsize=5, alpha=0.9, color=['tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:olive', 'tab:blue'])
    ax.set_title(metric)
    ax.set_ylabel("Average Value", fontsize=12)
    ax.set_ylim([min(means[i]) - max(stds[i]), max(means[i]) + max(stds[i])])  # Adjust y-limits
    ax.set_xticklabels(controllers, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout()

# plt.savefig('Figures/controller_comparison_energy_carbon_water.pdf', format='pdf')

plt.show()
#%%
