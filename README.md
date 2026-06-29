# FGENet: Frequency-Guided Enhancement Network for Infrared Small Target Detection

## Recommended Environment

- Python 3.11.7
- PyTorch 2.2.1
- torchvision 0.17.1

## Installation

```bash
  pip install -r requirements.txt
```
## Datasets
- **NUAA-SIRST** — [[Link]](https://github.com/YimianDai/sirst)
- **IRSTD-1K** — [[Link]](https://github.com/RuiZhang97/IRSTD-1K)
- **NUDT-SIRST** — [[Link]](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
Place datasets under `dataset/` with images/ and masks/ folders inside each dataset directory.
## Evaluate

```bash
  python eval.py --weight_nuaa ./checkpoint/FGENet_NUAA_SIRST_566_best.pth.tar
```
## Results
| Dataset       | mIoU     |  nIoU  | Pd       | Fa (×10⁶) |
|:------------- |:--------:|:------:|:--------:|:---------:|
| NUAA-SIRST    | 79.69%   | 80.37% | 97.72%   |   12.48   |
| IRSTD-1K      | 67.25%   | 68.95% | 92.93%   |   10.89   |
| NUDT-SIRST    | 92.35%   | 92.98% | 99.05%   |   2.45    |
Pre-trained weights: [[Baidu Pan]](https://pan.baidu.com/s/1BPf_LZ3kbGLaVYjGH5X_jw?pwd=zz12)
## Citation
If you find this work useful, please consider citing:
```bibtex

@article{,
  title={FGENet: Frequency-Guided Enhancement Network for Infrared Small Target Detection},
  author={},
  journal={},
  year={2026}
}

```
## Acknowledgement
- Thanks to [MSHNet](https://github.com/ying-fu/MSHNet) for the SLSLoss.
- Thanks to [SCTransNet](https://github.com/xdFai/SCTransNet) for the codebase. (Shuai Yuan)
- Thanks to [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) for the repository style. (Boyang Li)
