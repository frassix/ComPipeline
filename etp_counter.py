import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

def parse_etp(filename):
    """
    Parse an .etp file.

    Parameters
    ----------
    filename : str
        Path to the input .etp file.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists containing the Monte Carlo labels (MC), predicted event
        types (ET), and classification probabilities (TP).
    """

    mc = []
    et = []
    prob = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts)<2:
                continue

            if parts[0] == "MC":
                mc.append(parts[1])

            elif parts[0] == "ET":
                et.append(parts[1])
            
            elif parts[0] == "TP":
                prob.append(parts[1])

        return mc, et, prob
    
def compare(mc, et):
    """
    Compare Monte Carlo truth labels with predicted event types.
    
    Parameters
    ----------
    mc : list[str]
        List of Monte Carlo truth labels.
    et : list[str]
        List of predicted event types.

    Returns
    -------
    tuple
        Counts of MC and predicted events, total number of correctly
        classified events, per-class match counts, and a confusion matrix
        for COMP/PAIR classification.
    """

    # Background files do not contain MC labels
    if mc == []:
        print("[WARNING] No MC labels found in the file. Running without --debug? "
              "Match counts and confusion matrix will not be meaningful.")
        mc = ["BACK"] * len(et)

    counts_mc = {}
    for item in mc:
        if item in counts_mc:
            counts_mc[item] += 1
        else:
            counts_mc[item] = 1
    counts_mc = dict(sorted(counts_mc.items(), key=lambda x: x[1], reverse=True))

    counts_et = {}
    for item in et:
        if item in counts_et:
            counts_et[item] += 1
        else:
            counts_et[item] = 1
    counts_et = dict(sorted(counts_et.items(), key=lambda x: x[1], reverse=True))

    # Mapping from MC labels to classifier labels
    mapping = {
            'COMP': 'CO',
            'PAIR': 'PA',
            'PHOT': 'PH',
            'RAYL': 'RA',
            'BACK': 'MU'
        }
    
    # Number of correctly classified events by interaction type
    match_types = {
            'COMP': 0,
            'PAIR': 0,
            'PHOT': 0,
            'RAYL': 0,
            'BACK': 0
        }
    
    total_matches = 0

    # Confusion matrix for COMP vs PAIR

    cc = 0  # Actual COMP, Predicted CO
    cp = 0  # Actual COMP, Predicted PA
    pp = 0  # Actual PAIR, Predicted PA
    pc = 0  # Actual PAIR, Predicted CO

    for i in range(len(et)):
        true = mc[i]
        pred = et[i]
        if true in mapping and mapping[true] == pred:
            total_matches += 1
            match_types[true] += 1
        
        # Target matrix logic (COMP vs PAIR)
        if true == 'COMP':
            if pred == 'CO':
                cc += 1
            elif pred == 'PA':
                cp += 1
        elif true == 'PAIR':
            if pred == 'CO':
                pc += 1
            elif pred == 'PA':
                pp += 1

    match_types = dict(sorted(match_types.items(), key=lambda x: x[1], reverse=True))
    
    confusion_matrix = {
        'CC': cc,
        'CP': cp,
        'PC': pc,
        'PP': pp
    }

    confusion_matrix_pct = {
        'CC': (cc / (cc + cp)) * 100 if (cc + cp) > 0 else 0.0,
        'CP': (cp / (cc + cp)) * 100 if (cc + cp) > 0 else 0.0,
        'PC': (pc / (pc + pp)) * 100 if (pc + pp) > 0 else 0.0,
        'PP': (pp / (pc + pp)) * 100 if (pc + pp) > 0 else 0.0,
    }
    return counts_mc, counts_et, total_matches, match_types, confusion_matrix, confusion_matrix_pct

def print_n_save(counts_mc, counts_et, total_matches, match_types, cm, cmp, input_filename=None):
    """
    Print the analysis summary to the terminal and save it to a text file.
    """

    output_text = f"""
MC simulated events: {counts_mc}

Predicted event types: {counts_et}

Total matches: {total_matches}

Match type breakdown: {match_types}

-------------------------------------------------
               Predicted COMP      Predicted PAIR
Actual COMP  {cm['CC']:7.0f} - {cmp['CC']:3.2f}%    {cm['CP']:7.0f} - {cmp['CP']:3.2f}%
Actual PAIR  {cm['PC']:7.0f} - {cmp['PC']:3.2f}%    {cm['PP']:7.0f} - {cmp['PP']:3.2f}%
-------------------------------------------------
    """

    # 1. Print it to the console
    print(output_text)

    # 2. Save it to the txt file in the etp_counts folder
    output_dir = Path(__file__).resolve().parent / "etp_counts"
    output_dir.mkdir(exist_ok=True)

    if input_filename is not None:
        output_name = f"{Path(input_filename).stem}.txt"
    else:
        output_name = "counts.txt"

    output_path = output_dir / output_name
    with open(output_path, "w") as f:
        f.write(output_text)
    
    print(f"File successfully saved as {output_path}")

def probs_histo(et, prob):
    """
    Create a histogram of classifier probabilities.

    Only events classified as PA (pair production) or CO (Compton
    scattering) are included in the histogram.

    The figure is saved as 'probability_distribution.pdf'.
    """

    # Keep only probabilities associated with PA and CO predictions
    filtered_prob = [
        float(prob_val) if et_val in ['PA', 'CO'] else float('nan') 
        for et_val, prob_val in zip(et, prob)
    ]

    # Convert to a numpy array and filter out the NaNs
    data_array = np.array(filtered_prob)
    clean_data = data_array[~np.isnan(data_array)]

    # Histogram visualization
    plt.figure(figsize=(8, 5))
    plt.hist(clean_data, bins=20, color='C0', edgecolor='black', alpha=0.7)

    plt.title("Histogram of Probability Values (Filtered for PA & CO)", fontsize=14, fontweight='bold')
    plt.xlabel("Probability Values", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("probability_distribution.pdf")

def main():
    """
    Parse command-line arguments and run the analysis.
    """

    parser = argparse.ArgumentParser(
        description="Analyze an .etp event classification file."
    )
    parser.add_argument(
        "filename",
        help="Input .etp file."
    )
    parser.add_argument(
        "-p",
        "--probability",
        action="store_true",
        help="Generate and save the probability histogram."
    )

    args = parser.parse_args()

    try:
        mc, et, prob = parse_etp(args.filename)
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)
    
    # Counts and comparison
    counts_mc, counts_et, total_matches, match_types, cm, cmp = compare(mc, et)
    
    # Prints
    print_n_save(counts_mc, counts_et, total_matches, match_types, cm, cmp, input_filename=args.filename)

    # Generate the probability histogram only if requested.
    if args.probability:
        probs_histo(et, prob)
        print("\nProbability histogram saved as 'probability_distribution.pdf'")


if __name__ == "__main__":
    main()