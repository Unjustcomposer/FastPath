import matplotlib.pyplot as plt
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

def generate_comparison_table(our_metrics: Dict[str, Any], output_path: str) -> None:
    """
    Generates a formatted markdown table comparing against published baselines
    and a scatter plot of slides/hr vs AUC.
    
    Args:
        our_metrics: Dictionary containing our model's metrics with keys:
                     'AUC', 'F1', 'sensitivity', 'specificity', 
                     'ms_per_slide', 'slides_per_hour', 'params_millions'.
        output_path: Path to save the markdown table. The plot will be saved 
                     in the same directory.
    """
    baselines = [
        {"name": "CLAM (Lu 2021)", "auc": 0.968, "slides_per_hr": 60, "params": "25M"},
        {"name": "TransMIL (Shao 2021)", "auc": 0.963, "slides_per_hr": 40, "params": "42M"},
        {"name": "HDMIL (Dong 2025)", "auc": 0.971, "slides_per_hr": 84, "params": "30M"},
        {"name": "FastPath (Ours)", 
         "auc": our_metrics.get("AUC", 0.0), 
         "slides_per_hr": our_metrics.get("slides_per_hour", 0.0), 
         "params": f"{our_metrics.get('params_millions', 0.0)}M"}
    ]
    
    # Generate Markdown Table
    md_content = "# Baseline Comparison\n\n"
    md_content += "| Model | AUC | Slides / Hour | Parameters |\n"
    md_content += "|---|---|---|---|\n"
    
    for b in baselines:
        md_content += f"| {b['name']} | {b['auc']:.3f} | {b['slides_per_hr']} | {b['params']} |\n"
        
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Saved comparison table to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save markdown table: {e}")
        
    # Generate Scatter Plot
    plot_path = os.path.join(os.path.dirname(output_path), 'baseline_comparison.png')
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        for b in baselines:
            ax.scatter(b['slides_per_hr'], b['auc'], label=b['name'], s=150, alpha=0.8)
            ax.annotate(b['name'], (b['slides_per_hr'], b['auc']), 
                        xytext=(8, 5), textcoords='offset points')
            
        ax.set_xlabel("Slides per Hour")
        ax.set_ylabel("AUC")
        ax.set_title("Performance vs Speed Comparison")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved baseline comparison plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to generate baseline plot: {e}")
