"""
View all experimental results in a nice table
"""

import csv
import os
import sys

def view_results():
    results_file = "results.csv"

    if not os.path.exists(results_file):
        print("❌ No results yet. Run experiments first!")
        print("\nStart with:")
        print("  python 01_baseline_t5.py")
        return

    with open(results_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)

    if len(data) == 0:
        print("❌ Results file is empty!")
        return

    # Try to use tabulate, fallback to simple print
    try:
        from tabulate import tabulate

        print("\n" + "="*100)
        print("WIKIMULTIHOP COMPARISON RESULTS")
        print("="*100 + "\n")

        print(tabulate(data, headers=headers, tablefmt='grid'))

        # Calculate improvements over baseline
        if len(data) >= 2:
            baseline_em = float(data[0][4])

            print("\n" + "="*100)
            print("IMPROVEMENTS OVER BASELINE")
            print("="*100 + "\n")

            improvements = []
            for row in data[1:]:
                exp_name = row[0]
                em = float(row[4])
                improvement = em - baseline_em
                improvements.append([
                    exp_name,
                    row[2],  # Stage
                    row[3],  # Layer
                    f"{em:.2f}%",
                    f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
                ])

            print(tabulate(improvements,
                          headers=['Experiment', 'Stage', 'Layer', 'EM', 'Δ vs Baseline'],
                          tablefmt='grid'))

    except ImportError:
        # Fallback to simple print
        print("\n" + "="*100)
        print("WIKIMULTIHOP COMPARISON RESULTS")
        print("="*100 + "\n")

        # Print headers
        print(" | ".join(headers))
        print("-" * 100)

        # Print data
        for row in data:
            print(" | ".join(row))

        print("\n💡 Install tabulate for better formatting: pip install tabulate")

if __name__ == '__main__':
    view_results()
