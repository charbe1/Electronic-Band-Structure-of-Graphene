import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

#lattice parameters
a = 1.42  # Carbon-carbon bond length in Angstroms

#tight-binding hopping parameter
t = -2.7  # Nearest-neighbor hopping parameter (eV)

#nearest-neighbor vectors (connecting carbon atoms)
d1 = a * np.array([0, 1])
d2 = a * np.array([-np.sqrt(3) / 2, -1 / 2])
d3 = a * np.array([np.sqrt(3) / 2, -1 / 2])

#reciprocal lattice vectors
b1 = (2 * np.pi / (np.sqrt(3) * a)) * np.array([1, 1 / np.sqrt(3)])
b2 = (2 * np.pi / (np.sqrt(3) * a)) * np.array([1, -1 / np.sqrt(3)])

#high-symmetry points in the Brillouin zone
G_Gamma = 0 * b1 + 0 * b2
G_K = (1 / 3) * b1 + (1 / 3) * b2
G_M = (1 / 2) * b1 + 0 * b2

#phase factors for nearest-neighbor interactions
def f(k):
    return np.exp(1j * np.dot(k, d1)) + np.exp(1j * np.dot(k, d2)) + np.exp(1j * np.dot(k, d3))

#Hamiltonian (for pi band)
def H(k):
    return np.array([
        [0, t * f(k)],
        [t * f(k).conj(), 0]
    ])

#k-points along Gamma-K-M-Gamma path (increased for smoother curves)
k_values_Gamma_K = np.linspace(G_Gamma, G_K, 200)
k_values_K_M = np.linspace(G_K, G_M, 200)
k_values_M_Gamma = np.linspace(G_M, G_Gamma, 200)

k_values = np.concatenate((k_values_Gamma_K, k_values_K_M, k_values_M_Gamma))
#calculate the distances
k_distances = np.insert(np.cumsum(np.linalg.norm(np.diff(k_values, axis=0), axis=1)), 0, 0)

#calculate eigenvalues
eigenvalues = []
for k in k_values:
    evals = eigvals(H(k))
    eigenvalues.append(evals)


energies_pi = [eigval[0].real for eigval in eigenvalues]
energies_pi_star = [eigval[1].real for eigval in eigenvalues]

plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
plt.plot(k_distances, energies_pi, 'k', linewidth=2, label="π Band")
plt.plot(k_distances, energies_pi_star, 'r', linewidth=2, label="π* Band")


plt.axvline(k_distances[199], color='gray', linestyle='dashed', linewidth=1)  # K point
plt.text(k_distances[199], 0, 'K', ha='center', va='bottom')
plt.axvline(k_distances[399], color='gray', linestyle='dashed', linewidth=1)  # M point
plt.text(k_distances[399], 0, 'M', ha='center', va='bottom')
plt.axvline(k_distances[-1], color='gray', linestyle='dashed', linewidth=1)  # Gamma point
plt.text(k_distances[-1], 0, 'Γ', ha='center', va='bottom')
plt.axhline(0, color="k", linestyle="dashed", linewidth=1, label="Fermi Energy")
plt.xlabel("Distance in Brillouin Zone", fontsize=12)
plt.ylabel("Energy (eV)", fontsize=12)
plt.title("Graphene Band Structure (π and π*)", fontsize=14)
plt.ylim(-4, 4)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
