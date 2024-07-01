# HistoFeatEx

## Overview

HistoFeatEx is a tool for extracting patch-level features from histopathology images using various pretrained models. Implemented models include:

- **ResNet-50<sup>1</sup>**: Pretrained on ImageNet.
- **CTransPath<sup>2</sup>**: A self-supervised learning (SSL) model pretrained on the TCGA dataset with 14,325,848 unlabeled histopathological patches.
- **HistoSSL<sup>3</sup>**: An SSL model pretrained on the TCGA dataset with 43,374,755 unlabeled histopathological patches.
- **UNI<sup>4</sup>**: An SSL model pretrained on the Mass-100K dataset collected from Massachusetts General Hospital and Brigham and Women's Hospital with 100,130,900 patches extracted from 100,426 histology slides.
- **PLIP<sup>5</sup>**: A visual-language model, pretrained on 208,414 image-caption pairs.
- **CONCH<sup>6</sup>**: A visual-language model, pretrained on 1,786,362 image-caption pairs.

## Requirements

- Python 3.10
- PyTorch 2.3
- Torchvision 0.18
- Transformers 4.31 

## Installation

```bash
git clone https://github.com/ezgiogulmus/HistoFeatEx.git
cd HistoFeatEx
conda create -n histofex python=3.10 -y
conda activate histofex 
pip install -e .
```

For the installation of the CTransPath environment, please refer to their repository as it involves a modified `timm` package:
[TransPath GitHub Repository](https://github.com/Xiyue-Wang/TransPath)

The model weights for UNI and CONCH models can be accessed at Hugging Face repositories after request is confirmed. For CTransPath, it can be downloaded from their GitHub repository. Other model weights will be downloaded automatically within the code.

## Usage

First, extract patch coordinates using the [CLAM library](https://github.com/mahmoodlab/CLAM)<sup>7</sup>.
Then, run the command below to extract patch features.

```bash
python main.py --model_type uni --target_patch_size 224 --ckpt_path path/to/checkpoints --data_root_dir path/to/root/dir --data_h5_dir name_of_the_h5_folder --data_slide_dir name_of_the_slide_folder --feat_dir name_of_the_feat_folder --csv_file name_of_the_csv_file.csv
```

**Arguments:**
- `model_type`: `uni`, `conch`, `plip`, `ssl`, `ctp`, `resnet50`
- `ckpt_path`: Needed for UNI, CONCH, and CTP models
- `data_root_dir`: Root directory containing the data
- `data_h5_dir`: Directory containing patch coordinates and masks
- `data_slide_dir`: Directory containing SVS files
- `data_feat_dir`: Directory where extracted features are saved
- `csv_file`: CSV file with the list of slide_id and case_id 

**Folder Structure:**
```
data_root_dir
|
|__ data_slide_dir (where SVS files are located)
|__ data_h5_dir (where patch coordinates and masks are located)
|     |__ patches
|     |__ masks
|__ data_feat_dir (where extracted features are saved)
|__ slide_list.csv
```
## License

This repository is licensed under the [GPLv3](LICENSE) and is available for non-commercial academic purposes.

### Acknowledgements and Third-Party Licenses

- This code is adapted from the [CLAM repository](https://github.com/mahmoodlab/CLAM). Please refer to the CLAM repository for its license terms.
- The [CTransPath model](./models/ctran.py) is copied from the [TransPath repository](https://github.com/Xiyue-Wang/TransPath). Please refer to the TransPath repository for its license terms.
- The models HistoSSL, UNI, PLIP, and CONCH are used according to their respective licenses. Please refer to their respective repositories for license terms:
  - [HistoSSL](https://github.com/owkin/HistoSSLscaling)
  - [UNI](https://github.com/mahmoodlab/UNI)
  - [PLIP](https://github.com/PathologyFoundation/plip)
  - [CONCH](https://github.com/mahmoodlab/CONCH)

## References

1. He, Kaiming, et al. "Deep Residual Learning for Image Recognition." arXiv, 2015, eprint:1512.03385, arXiv, cs.CV.

2. Wang, Xiyue, et al. "Transformer-based Unsupervised Contrastive Learning for Histopathological Image Classification." *Medical Image Analysis*, Elsevier, 2022.

3. Filiot, Alexandre, et al. "Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling." *medRxiv*, Cold Spring Harbor Laboratory Press, 2023, doi:10.1101/2023.07.21.23292757.

4. Chen, Richard J., et al. "Towards a General-Purpose Foundation Model for Computational Pathology." *Nature Medicine*, Nature Publishing Group, 2024.

5. Huang, Zhi, et al. "A Visual-Language Foundation Model for Pathology Image Analysis Using Medical Twitter." *Nature Medicine*, Nature Publishing Group US New York, 2023, pp. 1-10.

6. Lu, Ming Y., et al. "A Visual-Language Foundation Model for Computational Pathology." *Nature Medicine*, vol. 30, 2024, pp. 863-874, Nature Publishing Group.

7. Lu, Ming Y., et al. "Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images." *Nature Biomedical Engineering*, vol. 5, no. 6, 2021, pp. 555-570, Nature Publishing Group.
