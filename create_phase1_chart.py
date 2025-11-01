import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Baseline\nStacking', 'Baseline\nXGBoost', 'Phase 1\nImproved RF', 
          'Phase 1\nCatBoost', 'Phase 1\nHierarchical']
r2_vals = [0.6134, 0.6118, 0.6173, 0.6172, 0.7161]
colors = ['lightcoral', 'lightcoral', 'lightblue', 'lightblue', 'lightgreen']

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

bars = ax.barh(models, r2_vals, color=colors, edgecolor='black', linewidth=2)

ax.set_xlabel('Test R²', fontsize=14, fontweight='bold')
ax.set_title('Phase 1 Achievement: +10.27 Point Improvement', fontsize=16, fontweight='bold')

ax.axvline(0.6134, color='red', linestyle='--', linewidth=2, label='Baseline Best')
ax.axvline(0.70, color='green', linestyle=':', linewidth=2, label='Phase 1 Target')

ax.legend(fontsize=12)

for i, (model, val) in enumerate(zip(models, r2_vals)):
    ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

ax.set_xlim(0.55, 0.75)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Visualizations_Phase1/phase1_achievement_chart.png', dpi=300, bbox_inches='tight')
print('✅ Chart saved: Visualizations_Phase1/phase1_achievement_chart.png')
plt.close()

