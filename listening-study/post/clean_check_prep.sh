#!/bin/bash
set -e

# Arguments:
# $1: filepath to data, e.g., ../data/group#_##_MM.DD.YYYY/group#_MM.DD.YYYY.xlsx
# $2: filepath to save clean data to, e.g., ../data/group#_##_MM.DD.YYYY/clean.xlsx
# $3: filepath to save incomplete data to, e.g., ../data/group#_##_MM.DD.YYYY/incomplete.xlsx
# $4: filepath for participant map, e.g., ../data/participant_map.xlsx

# Clean data 
# echo "Cleaning..."
# python3 clean.py --in_file $1 --out_file $2 --incomplete_file $3 --participant_map ../data/participant_map.xlsx
# echo ""

# # Aggregate data
# echo "Collecting all data..."
cd ../data
# python3 collect_all_data.py
# echo ""

# Remove non-accepted data
echo "Removing non-accepted data..."
python3 remove_non_accepts.py
echo "Done."

# Check for suspicious results
echo "Checking data..."
cd ../post
python3 check.py --file $2
echo ""

# Plot demographic data
echo "Plotting..."
python3 plot_demo.py --input ../data/all/clean.xlsx
echo "Done."
echo ""

# Prepare mos ratings data
echo "Analyzing..."
python3 analyze.py
echo "Done."
echo ""

# Compute correlations
echo "Calculating correlations..."
python3 corr.py
echo "Done."
echo ""