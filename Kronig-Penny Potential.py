import numpy as np
import matplotlib.pyplot as plt

# Define the range for x
x = np.linspace(-np.pi, np.pi, 1000)
y = np.linspace(0.1, 10, 1000)       # Change the end of y range to 10
X, Y = np.meshgrid(x, y)

# Calculate Z values (no change here)
Z = np.cos(X) - np.cos(Y) - (2.63 / Y) * np.sin(Y)

# Create the contour plot (no change here)
plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=[0], colors='blue')

# Customize the plot (only ylim is changed)
plt.xlabel('k', fontsize=12)
plt.ylabel('κ', fontsize=12)
plt.title('Contour Plot of cos(k) = cos(κ) + (2.63/κ)sin(κ)', fontsize=14)
plt.grid(axis='both', linestyle='--')
plt.xlim(-np.pi, np.pi)
plt.ylim(0, 10)     # Adjust the y-axis limit to 10

plt.show()

