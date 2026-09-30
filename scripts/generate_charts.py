"""Generate visualization charts for FlipGate social media posts."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# Data from our results
models = ['bf16\n(HF)', 'AWQ\n(HF)', 'GPTQ\n(HF)', 'f16\n(llama.cpp)', 'q4_K_M\n(llama.cpp)']
accuracy = [34.0, 45.0, 39.0, 31.0, 36.5]
r_to_w_flips = [0, 13, 14, 24, 21]
w_to_r_flips = [0, 35, 24, 18, 26]

# Chart 1: Accuracy vs Flip Rates
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar chart for accuracy
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2E86AB', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Line chart for flip rates
ax2 = ax1.twinx()
flip_rates = [(r/(r+w))*100 if (r+w) > 0 else 0 for r, w in zip(r_to_w_flips, w_to_r_flips)]
line1 = ax2.plot(x, flip_rates, 'o-', color='#A23B72', linewidth=2.5, markersize=10, 
                 label='R→W Flip Rate (%)', zorder=5)

# Add value labels on line
for i, rate in enumerate(flip_rates):
    if rate > 0:
        ax2.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='#A23B72')

ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2E86AB')
ax2.set_ylabel('Right→Wrong Flip Rate (%)', fontsize=12, fontweight='bold', color='#A23B72')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 55)
ax2.set_ylim(0, 80)

# Title
plt.title('FlipGate: Accuracy vs Per-Item Flip Rates\n(Qwen2.5-3B-Instruct, 200 GSM8K items)', 
          fontsize=14, fontweight='bold', pad=20)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('accuracy_vs_flips.png', dpi=150, bbox_inches='tight')
print("✓ Generated accuracy_vs_flips.png")

# Chart 2: Flip Direction Breakdown
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, r_to_w_flips, width, label='Right→Wrong (regression)', 
               color='#E63946', alpha=0.8)
bars2 = ax.bar(x + width/2, w_to_r_flips, width, label='Wrong→Right (improvement)', 
               color='#06A77D', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Flipped Items', fontsize=12, fontweight='bold')
ax.set_title('Flip Direction Analysis\n(200 common items, bf16 baseline)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 45)

# Add annotation
ax.annotate('AWQ: More improvements\nthan regressions', 
            xy=(1, 37), fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('flip_directions.png', dpi=150, bbox_inches='tight')
print("✓ Generated flip_directions.png")

# Chart 3: Statistical Significance Summary
fig, ax = plt.subplots(figsize=(10, 5))

comparisons = ['AWQ\nvs bf16', 'GPTQ\nvs bf16', 'f16\nvs bf16', 'q4_K_M\nvs bf16']
p_values = [0.0024, 0.1443, 0.4404, 0.5596]
colors = ['#06A77D' if p < 0.05 else '#FFB703' for p in p_values]

bars = ax.bar(comparisons, p_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add significance threshold line
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance threshold (p=0.05)')

# Add value labels
for bar, p in zip(bars, p_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'p={p:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('McNemar Test p-value', fontsize=12, fontweight='bold')
ax.set_title('Statistical Significance of Flip Asymmetry\n(Only AWQ shows significant difference)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 0.7)
ax.legend(fontsize=11)

# Add annotation for significant result
ax.annotate('✓ Significant\n(p=0.0024)', 
            xy=(0, 0.0024), xytext=(1.5, 0.15),
            fontsize=11, fontweight='bold', color='#06A77D',
            arrowprops=dict(arrowstyle='->', color='#06A77D', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('statistical_significance.png', dpi=150, bbox_inches='tight')
print("✓ Generated statistical_significance.png")

# Chart 4: Engine vs Quantization Effect
fig, ax = plt.subplots(figsize=(9, 6))

effects = ['Engine Effect\n(HF→llama.cpp)', 'Quantization Effect\n(f16→q4_K_M)']
flip_rates = [12.0, 7.0]
colors = ['#8338EC', '#FB5607']

bars = ax.bar(effects, flip_rates, color=colors, alpha=0.8, width=0.5, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, rate in zip(bars, flip_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Flip Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Engine Choice Matters More Than Quantization\n(Controlled experiment with f16 baseline)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 16)

# Add annotation
ax.annotate('Engine effect is 1.7× larger\nthan quantization effect!', 
            xy=(0.5, 9.5), fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('engine_vs_quantization.png', dpi=150, bbox_inches='tight')
print("✓ Generated engine_vs_quantization.png")

# Chart 5: Hallucination Detection
fig, ax = plt.subplots(figsize=(10, 6))

models_hall = ['bf16', 'AWQ', 'GPTQ-Int4']
hallucination_rate = [18.1, 32.9, 25.8]  # 100 - no_hallucination_rate
colors = ['#06A77D', '#E63946', '#FFB703']

bars = ax.bar(models_hall, hallucination_rate, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, width=0.5)

# Add value labels
for bar, rate in zip(bars, hallucination_rate):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('FedProc Registry Hallucination Detection\n(155 real FAR/DFARS records)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 40)

# Add significance annotations
ax.annotate('AWQ: +14.8 pts\np=0.001 ✓', 
            xy=(1, 32.9), xytext=(1.8, 35),
            fontsize=10, fontweight='bold', color='#E63946',
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))

ax.annotate('GPTQ: +7.7 pts\np=0.038 ✓', 
            xy=(2, 25.8), xytext=(0.5, 30),
            fontsize=10, fontweight='bold', color='#FFB703',
            arrowprops=dict(arrowstyle='->', color='#FFB703', lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('hallucination_detection.png', dpi=150, bbox_inches='tight')
print("✓ Generated hallucination_detection.png")

print("\n✅ All 5 charts generated successfully!")
print("Files: accuracy_vs_flips.png, flip_directions.png, statistical_significance.png,")
print("       engine_vs_quantization.png, hallucination_detection.png")
