
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

labels = ["Ham", "Spam"]
counts = [4825, 747]

ax.bar(labels, counts)
ax.set_title("Distribution of Ham and Spam Messages")
ax.set_xlabel("Message Category")
ax.set_ylabel("Number of Messages")

for i, value in enumerate(counts):
    ax.text(i, value + 50, str(value), ha="center")

fig.tight_layout()
fig.savefig("fig2_label_distribution.png", dpi=300)
plt.close(fig)

print("SUCCESS: Chart saved as fig2_label_distribution.png")
