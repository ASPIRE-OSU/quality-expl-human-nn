# How Frequency Band Importance Affects Neural Network Predictions and Human Perception for Speech Quality Assessment
This repository contains the implementation of the paper _How Frequency Band Importance Affects Neural Network Predictions and Human Perception for Speech Quality Assessment_ as published in Interspeech 2026.

## File Organization
- `listening-study`: the data and scripts used for the human subject listening study corresponding to sections 3 and 4.4 in the paper. 
- `network-expl`: the data and scripts used for genearting explanations for three neural networks (MOSNet, DNSMOS, SCOREQ), corresponding to sections 2 and 4 in the paper. 

## TO DO
- [ ] The NN explanations require .pkl files that are too big to upload to Git, so write a tutorial on how to generate them
- [ ] Move IUCOSINE data to data folder and update paths
- [ ] Add results folder for images and delete others
- [ ] Check file paths

## Citation
If you use any part of this work, please cite us
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

