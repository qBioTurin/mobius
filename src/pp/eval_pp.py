##  Post-processing of evaluation results

import argparse
import os 
import pandas as pd 


if __name__ == "__main__":
    """score pesato per condensare N metriche di performance in una 
- script per calcolare overall metric date:
	- quali metriche considerare
	- i pesi associati (somma pesi = 1)"""
    parser = argparse.ArgumentParser(description="Post-process evaluation results to compute overall metrics.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file containing evaluation results.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file to save overall metrics.")
    parser.add_argument("--metrics", type=str, nargs='+', required=True, help="List of metric names to consider.")
    parser.add_argument("--weights", type=float, nargs='+', required=True, help="List of weights corresponding to the metrics.")
    args = parser.parse_args()

    if len(args.metrics) != len(args.weights):
        raise ValueError("The number of metrics must match the number of weights.")
    
    if not abs(sum(args.weights) - 1.0) < 1e-6:
        raise ValueError("The sum of weights must be equal to 1.")
    
    df = pd.read_csv(args.input_file)
    