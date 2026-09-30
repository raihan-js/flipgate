"""Generate a professional thumbnail/cover image for FlipGate."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure with dark background
fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(8, 7.8, 'FlipGate', fontsize=72, fontweight='bold', 
        ha='center', va='center', color='#00d9ff',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#00d9ff', linewidth=3))

# Subtitle
ax.text(8, 6.5, 'Statistical Release Gate for Quantized LLMs', 
        fontsize=28, ha='center', va='center', color='#e94560', fontweight='bold')

# Key metrics boxes
metrics = [
    ('5', 'Models', 'bf16, AWQ, GPTQ, f16, q4_K_M'),
    ('3', 'Datasets', 'GSM8K, IFEval, FedProc'),
    ('4,078', 'Items Evaluated', 'Per-item flip tracking'),
    ('117', 'Tests Passing', 'Production-ready'),
]

box_width = 3.2
box_height = 2.0
start_x = 1.0
y_pos = 3.5

for i, (value, label, desc) in enumerate(metrics):
    x_pos = start_x + i * (box_width + 0.3)
    
    # Box background
    box = FancyBboxPatch((x_pos, y_pos), box_width, box_height, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#16213e', edgecolor='#0f3460', linewidth=2)
    ax.add_patch(box)
    
    # Value
    ax.text(x_pos + box_width/2, y_pos + 1.4, value, 
            fontsize=42, fontweight='bold', ha='center', va='center', color='#00d9ff')
    
    # Label
    ax.text(x_pos + box_width/2, y_pos + 0.8, label, 
            fontsize=16, fontweight='bold', ha='center', va='center', color='#e94560')
    
    # Description
    ax.text(x_pos + box_width/2, y_pos + 0.3, desc, 
            fontsize=10, ha='center', va='center', color='#a8a8a8')

# Key finding callout
callout = FancyBboxPatch((2, 1.2), 12, 1.5, 
                         boxstyle="round,pad=0.2", 
                         facecolor='#e94560', edgecolor='#ff6b6b', linewidth=3, alpha=0.9)
ax.add_patch(callout)

ax.text(8, 1.95, 'KEY FINDING: AWQ improved accuracy by +11 pts but broke 13 correct answers', 
        fontsize=18, fontweight='bold', ha='center', va='center', color='white')

ax.text(8, 1.45, 'Aggregate metrics hide per-item regressions. FlipGate catches them.', 
        fontsize=14, ha='center', va='center', color='#ffe5e5')

# Footer
ax.text(8, 0.4, 'github.com/raihan-js/flipgate  •  huggingface.co/datasets/raihan-js/flipgate-results', 
        fontsize=12, ha='center', va='center', color='#a8a8a8', style='italic')

plt.tight_layout()
plt.savefig('flipgate_thumbnail.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Generated flipgate_thumbnail.png")
