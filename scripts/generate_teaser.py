"""
Generate a teaser motivation figure (Fig 1) for the MemoryAD paper.
Plots the I-AUROC on Task 1 (bottle, cable, capsule) after training on Task 1 vs after 5 tasks.
Compares: Naive (catastrophic forgetting) vs MemoryAD (near-zero forgetting).
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Generating teaser figure...")
    
    # Values extracted from the E1 MVTec 5-task runs
    # Mean I-AUROC on Task 1 (first 3 categories)
    
    # Right after Task 1
    naive_t1 = 94.6
    memoryad_t1 = 94.6
    
    # After Task 5
    naive_t5 = 56.4      # Catastrophic forgetting
    memoryad_t5 = 93.9   # Near-zero forgetting
    
    labels = ["Naive (Lower Bound)", "MemoryAD (Ours)"]
    t1_scores = [naive_t1, memoryad_t1]
    t5_scores = [naive_t5, memoryad_t5]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    rects1 = ax.bar(x - width/2, t1_scores, width, label='After Task 1', color='#9E9E9E', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, t5_scores, width, label='After Task 5', color=['#F44336', '#2196F3'], edgecolor='black', linewidth=1.2)
    
    # Add horizontal line for Joint (Upper Bound)
    joint_ub = 95.1
    ax.axhline(joint_ub, color='#4CAF50', linestyle='--', linewidth=1.5, label='Joint (Upper Bound)')
    
    # Formatting
    ax.set_ylabel('I-AUROC (%) on Task 1 Categories', fontsize=12, fontweight='bold')
    ax.set_title('Catastrophic Forgetting in Continual Anomaly Detection', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(40, 100)
    
    # Add values on top of bars
    def autolabel(rects, is_t5=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            color = 'black'
            weight = 'normal'
            if is_t5 and i == 1:
                weight = 'bold'
                
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, color=color, fontweight=weight)
                        
    autolabel(rects1)
    autolabel(rects2, is_t5=True)
    
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add forgetting magnitude arrow for Naive
    ax.annotate('', xy=(x[0] + width/2, naive_t5), xytext=(x[0] + width/2, naive_t1),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(x[0] + width/2 + 0.05, (naive_t1 + naive_t5)/2, '-38.2%\nForgetting', 
            color='red', va='center', fontweight='bold')
            
    # Add stability note for MemoryAD
    ax.text(x[1] + width/2, memoryad_t5 + 5, 'Maintains\nPerformance', 
            ha='center', color='#0D47A1', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    out_path = "paper/figures/f1_teaser.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
