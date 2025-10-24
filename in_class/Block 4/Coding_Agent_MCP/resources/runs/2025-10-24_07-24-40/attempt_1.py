import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Draw house
house = plt.Rectangle((0.3, 0.1), 0.4, 0.4, color='brown')
roof = plt.Polygon([[0.3, 0.5], [0.7, 0.5], [0.5, 0.7]], color='red')
ax.add_patch(house)
ax.add_patch(roof)

# Draw tree
tree_trunk = plt.Rectangle((0.8, 0.2), 0.05, 0.2, color='saddlebrown')
tree_foliage = plt.Circle((0.825, 0.5), 0.1, color='green')
ax.add_patch(tree_trunk)
ax.add_patch(tree_foliage)

# Draw sun
sun = plt.Circle((0.15, 0.85), 0.1, color='yellow')
ax.add_patch(sun)

# Set limits and aspect
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('plot.png')