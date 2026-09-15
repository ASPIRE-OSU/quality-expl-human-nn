# Listening Study for the Effects of Frequency Bands on Human Perception of Speech Quality
This folder contains the data and scripts for performing and analyzing the human listening study. This corresponds to sections 3 and 4.4 in [the paper](TODO). 

## File Structure
- `data/`: contains raw audio files and metadata for the listening study.
- `post/`: contains scripts for processing and analyzing the listening study data
- `results/`: contains output files for the scripts in `post`. 

## Usage
### Environment Setup
The virtual environments for the models used in `network-expl/` should be sufficient to run the following script without additional installations. Regardless, the required packages are as follows.
```
pip install pandas tqdm scikit-learn ast seaborn matplotlib argparse numpy soundfile librosa
```

### Data Organization
`data/src-files.csv` lists the eight original, unperturbed IUCOSINE files that were used in the listening study. These files were perturbed and organized into (original, perturbed) pairs for each perturbation scenario and then grouped into batches for the listening study; see `data/batches.csv`. `data/map_to_original_name.csv` maps the raw audio file name to the organizational name (`batchNumber_pairNumber_fileNumber.csv`) used in the listening study. 

The raw audio files for the perturbed files can be found in `../network-expl/data/perturbed_iucosine/`. 

### Analysis
The scripts to clean, analyze, and produce figures have been consolidated into a single bash script. More information on the individual scripts can be found in `post/README.md`.
```
cd post
chmod +x clean_check_prep.sh
./clean_check_prep.sh
```
All outputs are saved in `results/` with the exception of the correlation output which is saved to `post/`.