# vehicle_reid_by_parsing
This repo gives the code for the paper "Xinchen Liu, Wu Liu, Jinkai Zheng, Chenggang Yan, Tao Mei: [Beyond the Parts: 
Learning Multi-view Cross-part Correlation for Vehicle Re-identification](https://lxc86739795.github.io/papers/2020_ACMMM_PCRNet.pdf). ACM MM 2020".
This code is based on [reid strong baseline](https://github.com/michuanhaohao/reid-strong-baseline).

## Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.0
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`

## Data Preparation

To train a vehicle reid model with parsing, you need the original image datasets like [VeRi](https://github.com/JDAI-CV/VeRidataset) and the parsing masks of all images.
For vehicle parsing models pretrained on the [MVP dataset](https://lxc86739795.github.io/MVP.html) based on PSPNet/DeepLabV3/HRNet, please refer to this [repo](https://github.com/lxc86739795/human_vehicle_parsing_platform).

## Training

You can run the examplar training script in `.sh` files.

## Main Code

The main code for GCN can be found in 
```bash
root
  engine
    trainer_selfgcn.py    # training pipline
  modeling
    baseline_selfgcn.py   # definition of the model
  tools
    train_selfgcn.py      # training preparation

```

The code for data io and sampler also be modified for the parsing based reid method.

## License

PCRNet is released under the [Apache 2.0 license](LICENSE).

## Reference
```BibTeX

@inproceedings{mm/LiuLZY020,
  author    = {Xinchen Liu and
               Wu Liu and
               Jinkai Zheng and
               Chenggang Yan and
               Tao Mei},
  title     = {Beyond the Parts: Learning Multi-view Cross-part Correlation for Vehicle
               Re-identification},
  booktitle = {ACM MM},
  pages     = {907--915},
  year      = {2020}
}
```
