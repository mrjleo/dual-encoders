# Dual-Encoders for IR
This code allows training of dual-encoder models for retrieval or ranking. The query and document encoder are the same model and share weights.

**Documentation and readme are incomplete.**

## Environment
You can use the file `conda_env.txt` to create a conda environment with the required packages:
```
conda create -n myenv --file conda_env.txt -c conda-forge -c pytorch
```

Afterwards, install the following:
- [ranking-utils](https://github.com/mrjleo/ranking-utils)

## Data Pre-Processing
Data pre-processing is done using the ranking-utils library:
```
python -m ranking_utils.scripts.create_h5_data \  
    dataset=ms_marco_passage \
    dataset.root_dir=/path/to/dataset/files \
    hydra.run.dir=/path/to/output/files
```
