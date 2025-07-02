import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Interactieve FP / FN plot met threshold, positieve samples & score distributie")

n_samples = 1000

# Sliders
pos_percentage = st.slider("Percentage positieve samples (%)", 1, 50, 10, 1)
threshold = st.slider("Threshold", 0.0, 1.0, 0.95, 0.01)
separation = st.slider("Score scheiding (0 = veel overlap, 1 = perfecte scheiding)", 0.0, 1.0, 0.7, 0.01)

# Labels
np.random.seed(42)
labels = np.zeros(n_samples)
n_pos = int(n_samples * pos_percentage / 100)
labels[:n_pos] = 1

# Scores genereren met controlled separation:
# Negatieven in [0, 0.5 * (1 - separation)] (dus bij hoge separation laag)
# Positieven in [0.5 + 0.5 * separation, 1] (bij hoge separation hoog)
neg_max = 0.5 * (1 - separation)
pos_min = 0.5 + 0.5 * separation

scores = np.zeros(n_samples)
scores[labels == 0] = np.random.uniform(0, neg_max, n_samples - n_pos)
scores[labels == 1] = np.random.uniform(pos_min, 1, n_pos)

# Precompute FP en FN over thresholds
thresholds = np.linspace(0, 1, 100)
fps = []
fns = []
for t in thresholds:
    preds = (scores >= t).astype(int)
    fps.append(np.sum((preds == 1) & (labels == 0)))
    fns.append(np.sum((preds == 0) & (labels == 1)))
fps = np.array(fps)
fns = np.array(fns)

# Huidige FP en FN bij geselecteerde threshold
current_fp = np.sum((scores >= threshold) & (labels == 0))
current_fn = np.sum((scores < threshold) & (labels == 1))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, fps, label='False Positives (FP)', color='red')
ax.plot(thresholds, fns, label='False Negatives (FN)', color='blue')
ax.axvline(threshold, color='gray', linestyle='--', label=f'Threshold = {threshold:.2f}')
ax.scatter([threshold], [current_fp], color='red')
ax.scatter([threshold], [current_fn], color='blue')

ax.set_xlabel("Threshold")
ax.set_ylabel("Aantal")
ax.set_title("Effect van threshold op FP en FN")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.write(f"### Bij threshold = {threshold:.2f}, {pos_percentage}% positieve samples, scheiding = {separation:.2f}:")
st.write(f"- False Positives: {current_fp}")
st.write(f"- False Negatives: {current_fn}")
