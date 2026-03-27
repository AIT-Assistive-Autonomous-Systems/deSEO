# deSEO: Physics-Aware Dataset Creation for High Resolution Satellite Images Shadow Removal

## Authors
**Lorenzo Beltrame<sup>1,4</sup>, Jules Salzinger<sup>1</sup>, Filip Svoboda<sup>2</sup>, Phillipp Fanta-Jende<sup>1</sup>, Jasmin Lampert<sup>1</sup>, Radu Timofte<sup>3</sup>, Marco Körner<sup>4</sup>**

<sup>1</sup> Austrian Institute of Technology, Vienna, Austria  
<sup>2</sup> University of Cambridge, Cambridge, UK  
<sup>3</sup> University of Würzburg, Würzburg, Germany  
<sup>4</sup> Technical University of Munich, Munich, Germany  

Contacts: name.surname@ait.ac.at or name.surname@tum.de

**TL;DR**
This demo currently provides:
- Scripts to construct deSEO training pairs from the S-EO dataset
- Devcontainer configuration for a reproducible environment
- Pipelines to replicate the experiments reported in the paper


The trained weights will be released soon.

## Before getting started
Download the devcontainer extension for VSCode and build the docker image contained in .devcontainer. Specify your clearml credentials or delete them, if you are not using them. If you want to use a mounting point, add it into the devcontainer.json file.

## DATASET: S-EO

The S-EO dataset is part of the publication:
"S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications"

Download Instructions:

1. Download the dataset from [https://huggingface.co/datasets/emasquil/shadow-eo](https://huggingface.co/datasets/emasquil/shadow-eo)
2. Move the dataset into the "datasets/" directory at the root of the project (create it if necessary).
3. (Optional) Rename the folder to SEO.
4. Unzip each folder and store the data locally.

## PREPARING THE SHADOW REMOVAL DATASET FROM S-EO

1. Install the required packages (or build the Docker image when available).
2. Set up the environment variable in your terminal:
   ```
   export LD\_LIBRARY\_PATH="\$CONDA\_PREFIX/lib\${LD\_LIBRARY\_PATH:+:}\$LD\_LIBRARY\_PATH"
   ```
3. Build noisy/weak ground truth pairs:
   ```
   python -m data\_management.build\_dataset dataset=deSEO\_better\_shadow\_filtering\_winter\_filtering\_v3
   ```
4. Create a local cache of preprocessed pairs:
   ```
   python -m data\_management.offline\_align\_cache
   ```

The dataset is now ready for training or inference.

## TRAINING THE SHADOW REMOVAL MODULE

The current demo version of this repository includes only the dataset construction pipeline.
Training and inference scripts, the ablation study, along with pretrained weights and Docker images for full reproducibility, will be released soon.

## ACKNOWLEDGMENTS
This work is inspired by:
CycleGAN and pix2pix in PyTorch: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

It is also partially based on the code infrastructure from:
S. Luo, H. Li\*, Y. Li, C. Shao, H. Shen, and L. Zhang,
"An Evolutionary Shadow Correction Network and a Benchmark UAV Dataset for Remote Sensing Images,"
IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–14, 2023, Art no. 5615414.


## Final note

If you find our work to be interesting please cite it:

```bibtex
@article{beltrame2026deseo,
  author  = {Lorenzo Beltrame and Jules Salzinger and Filip Svoboda and
             Phillipp Fanta-Jende and Jasmin Lampert and Radu Timofte and
             Marco K{\"o}rner},
  title   = {deSEO: Physic-Aware Dataset Creation for High Resolution Satellite Images Shadow Removal},
  journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year    = {2026},
  note    = {To appear},
  doi     = {TBD}
}

```
