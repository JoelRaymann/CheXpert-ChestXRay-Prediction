# for python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import argparse, sys, traceback

# Import in-house packages
import preprocessing_utils

VERSION = preprocessing_utils.__version__
DESCRIPTION = """
This is an API to Clean the CheXpert Dataset's metadata csv
for the Chest XRay Classification

Example Call would be
python clean_data.py -i <input metadata csv> -o <output metadata csv> -n <nan value to fill>
"""

if __name__ == "__main__":

    # For help
    parser = argparse.ArgumentParser(description = DESCRIPTION)

    # Add options
    parser.add_argument("-v", "--version", help = "Shows program version", action = "store_true")
    parser.add_argument("-i", "--input-csv", help = "Input Dataset's csv metadata to clean")
    parser.add_argument("-o", "--save-output", help = "The output csv path to store the data")
    parser.add_argument("-n", "--nan-value", help = "The value to substitute instead of NaN")

    # Read args
    args = parser.parse_args()

    # check for version
    if args.version:
        print("Using Version %s" %(VERSION))
        sys.exit(2)
    
    dataset_path = ""
    save_path = ""
    nan_value = 0
    if not args.input_csv or not args.save_output or not args.nan_value:
        print("{CRITICAL} Invalid arguments: please use help to know how to use")
        sys.exit(2)

    if args.input_csv:
        dataset_path = str(args.input_csv)
    if args.save_output:
        save_path = str(args.save_output)
    if args.nan_value:
        nan_value = int(args.nan_value)
    
    print("[INFO]: Cleaning Dataset in %s and saving in %s. Filling NaN with %d value: " %(dataset_path, save_path, nan_value))
    preprocessing_utils.clean_data(dataset_path, save_path, nan_value)
    print("[INFO]: Done")