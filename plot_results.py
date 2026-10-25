"""
Reproduce Figure 1 (Main Waiting Time by Algorithm) and Figure 2 (CWABP vs EA RMSE)
from simulation results, matching the exact styling in paper/paper.pdf.
"""
from __future__ import annotations

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PAPER_DIR = os.path.join(REPO_ROOT, "paper")

def plot_figure_1():
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print(f"Cannot find {summary_path}. Run `python run_experiment.py` first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    algs = ["A", "B", "C", "D", "E", "F"]
    avg_waits = [summary[a]["avg_waiting"] for a in algs]
    max_waits = [summary[a]["max_waiting"] for a in algs]

    x = np.arange(len(algs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    rects1 = ax.bar(x - width/2, avg_waits, width, label='Avg. waiting', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, max_waits, width, label='Max. waiting', color='#d95f02')

    ax.set_ylabel('Time units', fontsize=9)
    ax.set_xlabel('Algorithm', fontsize=9)
    ax.set_title('Figure 1: Mean Waiting Times across 65 Test Workloads', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algs, fontsize=9)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_pdf = os.path.join(PAPER_DIR, "fig_main_waiting.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"Generated {out_pdf}")

def plot_figure_2():
    csv_path = os.path.join(RESULTS_DIR, "cwabp_results.csv")
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Run `python run_experiment.py --cwabp-only` first.")
        return

    categories = []
    ea_rmse = []
    cw_rmse = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat_name = row["category"].replace("_", " ").replace("driven", "\ndriven")
            categories.append(cat_name)
            ea_rmse.append(float(row["ea_rmse"]))
            cw_rmse.append(float(row["cwabp_rmse"]))

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - width/2, ea_rmse, width, label='Exp. averaging', color='#7570b3')
    ax.bar(x + width/2, cw_rmse, width, label='CWABP', color='#1b9e77')

    ax.set_ylabel('Prediction RMSE', fontsize=9)
    ax.set_title('Figure 2: Burst Prediction RMSE (Exp. Averaging vs. CWABP)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_pdf = os.path.join(PAPER_DIR, "fig_cwabp_rmse.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"Generated {out_pdf}")

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
