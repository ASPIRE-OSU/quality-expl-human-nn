# Neural Network Explanations for Speech Quality Assessment
This folder contains the data and scripts for generating explanations for the DNSMOS, MOSNet, and SCOREQ models based on their input features (frequency bands). This corresponds to sections 2 and 4 in [the paper](TODO). 

## File Structure
- `data/`: directory containing perturbed audio files for the perturbation study as well as some aggregated data files
- `dnsmos/`: directory containing scripts for the DNSMOS network
- `mosnet/`: directory containing scripts for the MOSNet network
- `omnixai/`: the OmniXAI explanation framework
- `scoreq/`: directory containing scripts for the SCOREQ network
- `pick_files.py`: a script used to check which files have produced explanations and picks a specified number that have no explanations. This is especially helpful when using the `-s` argument (see [Generating SHAP Explanations](#Generating-SHAP-Explanations)). 
- `plot-freq-vs-shap.py	`: a script to plot SHAP and PDP results side-by-side à la Figure 2 in the paper.
- `tabular_regression_freqfeat.py`: the main script for producing explanations, either local or global. This script is a framework that based on arguments (specifying type of explanation, model, input files, etc.) calls the correct model/explanation function. See below for usage examples.

## Usage
### Environment Setup
We recommend creating a virtual environment for each network, following the original authors' requirements. In addition to the networks' requirements, you will need the following libraries. 
```
pip install argparse pandas numpy tqdm scikit-learn matplotlib scipy ast mpl_axes_aligner soundfile librosa onnxruntime tensorflow urllib shap
```

### Data Preparation
See `../README.md` for more information on the dataset. To summarize, we use a subset of [the IUB dataset](https://huggingface.co/datasets/aspire-osu/iub-dataset). The files from this dataset containing the substring 'COSINE' must be downloaded. Our scripts assume that they are downloaded into the data folder: `../data/MOS_Quality_INTERSPEECH_2020/16k_speech`. If you save the audio files to a different folder location, be sure to update the global path variables:
- `tabular_regression_freqfeat.py`: line 28
- `dnsmos/dnsmos_tabregression_freqfeats.py`: line 18
- `mosnet/mosnet_tabregression_freqfeats.py`: line 33
- `scoreq/dnsmos_tabregression_freqfeats.py`: line 33

### Generating SHAP Explanations
The following command produces local (SHAP) explanations that are saved to `./[model name]/results/shap`. The recommended maximum memory limit for running SHAP explanations is 115000 MB. 
```
python3 tabular_regression_freqfeat.py -m [model] -le -n 10 -r [int]
```
where 
- `-m [model]` (`--model [model]`) is one of `MOSNet`, `DNSMOS`, or `SCOREQ`
- `-le` (`--local_exp`) indicates that we are generating a local (i.e. SHAP) explanation
- `-n 10` (`--neighborhood 10`) defines the neighborhood size when computing SHAP values. Larger neighborhoods will result in, possibly more accurate, but more computationally expensive runs. A neighborhood of 10 was used in our paper. This argument is optional and will default to 1. 
- `-r [int]` (`--number [int]`) defines the number of files to generate local explanations for, selected randomly. Conversely, the option `-s [space separated list of audio paths]` (`--source`)can be used to generate a local explanation for specific file(s). 

Other command line options are:
- `-p`/`--prep_data`: if present, the script will perform data preprocessing tasks to ensure the audio files have been processed and in are in both the input format of the model as well as a format that the explanation framework can process. This is one of the most computationally and memory expensive steps and should be performed only once. This option will store the preprocessed data in a `.pkl` file that can be loaded on subsequent runs. **This option should be used on the first run to format the data correctly, but does not need to be used on subsequent runs.**
- `-s [space separated list of audio file paths]`/`--source [space separated list of audio file paths]` allows the user to specify specific file(s) for which to generate an explanation. Note that this option is the converse to `-r` which randomly selects a specified number of files. `-s` and `-r` should not be used in the same run. If both options are present, `-r` will take precedence. 
- `-i [int]`/`--index [int]` is similar to `-s` in that it allows the user to specify a specific file to use when generating explanations. The provided integer references the index of the file to use within the preprocessed data. 
- `-mbo [file path]`/`--bak_ovr_model_path [file path]` is an optional argument that specifies the path of the `.onnx` file to use when using DNSMOS. This defaults to DNSMOS's original `bak_ovr.onnx` file provided by the authors. This argument is predominantly useful if DNSMOS has been retrained.
- `-l [int]`/`--input_length [int]` is an optional argument used to specify the input file length when using DNSMOS. It defaults to 9.

### Generating PDP Explanations
The following command produces global (PDP) explanations that are saved to `./[model name]/results/pdp`. PDP explanations are memory-intensive to produce since they use the entire dataset. We recommend allowing a maximum memory of 23000 MB. 
```
python3 tabular_regression_freqfeat.py -m [model] -ge -f [space separated list of feature names]
```
where
- `-m [model]` (`--model [model]`) is one of `MOSNet`, `DNSMOS`, or `SCOREQ`
- `-ge` (`--global_exp`) indicates that we are generating a global (i.e. PDP) explanation
- `-f [space separated list of feature names]`/`--features [space separated list of feature names]` is a list of target features to explain. A separate PDP explanation is generated for each feature in this list. The feature names should match the column names from the preprocessed data. For this work, the feature names have the format "freqbin_" followed by the band number in the range [0, 160]. 

### Creating Plots from Figure 2
Once SHAP and PDP results have been generated, the explanations can be plotted side by side, matching Figure 2 from the paper. 
```
python3 plot-freq-vs-shap.py -m [model name]
```
where `-m [model name]` (or `--model [model name]`) is one of `MOSNet`, `DNSMOS`, or `SCOREQ`. Note that this script assumes the SHAP and PDP explanations are located in the model's folder under `results/shap` and `results/pdp`, respectively. Produced images are located in the SHAP results folder and aggregated PDP is output to a CSV in the PDP results folder. 