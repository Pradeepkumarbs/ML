import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Create sample n-dimensional data
np.random.seed(0)
data = pd.DataFrame({
    'A': np.random.rand(50),
    'B': np.random.rand(50),
    'C': np.random.rand(50),
    'D': np.random.rand(50)
})

# -----------------------------
# 1. Scatter Plot
# -----------------------------
plt.figure()
plt.scatter(data['A'], data['B'])
plt.title("Scatter Plot (A vs B)")
plt.xlabel("A")
plt.ylabel("B")
plt.show()

# -----------------------------
# 2. Box Plot
# -----------------------------
plt.figure()
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()

# -----------------------------
# 3. Heatmap (Correlation)
# -----------------------------
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Heatmap")
plt.show()

# -----------------------------
# 4. Contour Plot
# -----------------------------
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

plt.figure()
plt.contour(x, y, z)
plt.title("Contour Plot")
plt.show()

# -----------------------------
# 5. 3D Surface Plot
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)

ax.set_title("3D Surface Plot")
plt.show()
