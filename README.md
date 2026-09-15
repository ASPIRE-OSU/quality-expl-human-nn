# How Frequency Band Importance Affects Neural Network Predictions and Human Perception for Speech Quality Assessment
This repository contains the implementation of the paper _How Frequency Band Importance Affects Neural Network Predictions and Human Perception for Speech Quality Assessment_ as published in Interspeech 2026.

In this work, we used several explanation approaches (Shapley Additive Explanations, partial dependence plots, and perturbation analysis) to determine which frequency bands were most influential for three speech quality assessment neural networks: DNSMOS, MOSNet, and SCOREQ. See `network-expl` for more information. 

Following the network-based study, we performed a small-scale listening study to determine if human perception was influenced by the same factors as these deep neural networks. See `listening-study` for more information. 

## File Organization
- `listening-study`: the data and scripts used for the human subject listening study corresponding to sections 3 and 4.4 in the paper. 
- `network-expl`: the data and scripts used for generating explanations for three neural networks (MOSNet, DNSMOS, SCOREQ), corresponding to sections 2 and 4 in the paper. 

## Quick Start
See the READMEs in `listening-study/` and `network-expl/` for tutorials. 

## Data
For our experiments, we use a portion of [Dong and Williamson's IUB dataset](https://www.isca-archive.org/interspeech_2020/dong20_interspeech.html). This dataset contains mean opinion scores (MOS) from five human listeners for 18,000 audio files from the [COnversational Speech In Noisy Environments (COSINE) corpus](https://ieeexplore.ieee.org/document/4960543) and 18,000 files from the [Voices Obscured in Complex Environmental Settings](https://iqtlabs.github.io/voices/) dataset. This dataset is available [here](https://huggingface.co/datasets/aspire-osu/iub-dataset).

Our work uses only the COSINE files from IUB, which we call IUCOSINE. The file names and their labels are described in `data/iucosine.csv`.

## Citation
If you use any part of this work, please cite us.
```
@inproceedings{quality-expl-human-nn,
    authors = {Lamba, Ada and Williamson, Donald S.},
    title = {{How Frequency Band Importance Affects Neural Network Predictions and Human Perception for Speech Quality Assessment}},
    month = {09},
    year = {2026},
    city = {Sydney, Australia}
    booktitle = {Proc. Interspeech}
}
```

