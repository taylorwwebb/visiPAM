# Zero-shot Visual Reasoning through Probabilistic Analogical Mapping

Analysis code and behavioral data for the paper [Zero-shot Visual Reasoning through Probabilistic Analogical Mapping](https://arxiv.org/abs/2209.15087). 

The following command will reproduce Figure 5 (comparing human 3D object mapping with visiPAM), and perform repeated-measures ANOVA:
```
python3 ./analysis.py
```
The corresponding data can be found in ```human_visiPAM_data.csv```.

To perform the version of the analysis in which human behavioral responses are clustered (to account for bimodal response distributions), run:
```
python3 ./analysis.py --diptest
```

To perform correlation analysis (item-level correlation between human and visiPAM responses), run:
```
python3 ./correlation_analyses.py
```
To perform correlation analysis with the node-only visiPAM ablation model, run:
```
python3 ./correlation_analyses.py --model node_only
```
To perform correlation analysis with the edge-only visiPAM ablation model, run:
```
python3 ./correlation_analyses.py --model edge_only
```

Raw human behavioral data (including marker placement for each trial) can be accessed in the file ```raw_data.human.xlsx```.

Behavioral stimuli can be found in the directory ```./stimuli```.

Pointcloud data can be found in the directory ```./pointclouds```.

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [pingouin](https://pingouin-stats.org/build/html/index.html)

## Authorship

All code was written by [Taylor Webb](https://github.com/taylorwwebb) and [Shuhao Fu](https://github.com/fushuhao6). 
