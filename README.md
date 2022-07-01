# Constructions in Combinatorics via Neural Networks
Code accompanying the bachelor thesis "Constructions in Combinatorics via Neural Networks", written by Colin Doumont and supervised by Lars Rohwedder.

## Software requirements
See requirements.txt file for necessary Python packages.

## Files and usage
1. basic.py: basic adaption of Wagner's code for discrepancy theory
2. parallel.py: identical to basic.py, except it runs in parallel on all available CPUs
3. fractional.py: more sophisticated adaptation of Wagner's code, now also working for fractional discrepancy
4. discrepancy.py: discrepacy-related functions used in files 1, 2 and 3
5. dynamic.py: dynamic program used in files 1 and 2
6. random_search.py: naive approach to finding matrices with a certain discrepancy
7. NN_search.py: identical to parallel.py, except it runs for 100 trials and then logs the average
8. plotting.py: plots for the figures in the paper