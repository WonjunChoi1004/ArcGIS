import numpy as np
import matplotlib.pyplot as plt

# Parameters for normal distribution of soil cohesion (c in kPa)
mean_c = 25
std_dev_c = 5

# Generate x values
c_values = np.linspace(mean_c - 4 * std_dev_c, mean_c + 4 * std_dev_c, 500)
pdf_values = (1 / (std_dev_c * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((c_values - mean_c) / std_dev_c) ** 2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(c_values, pdf_values, color='blue', linewidth=4)
plt.title('Probability Distribution of Soil Cohesion (c)', fontsize=22)
plt.xlabel('Cohesion c (kPa)', fontsize=18)
plt.ylabel('Probability Density', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
