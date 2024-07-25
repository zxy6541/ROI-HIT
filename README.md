# ROI-HIT: Region of Interest-driven High-dimensional Microarchitecture Design Space Exploration
ROI-HIT is a Region Of Interest (ROI)-driven high-dimensional microarchitecture Design Space Exploration (DSE) method. By focusing on the promising ROIs, ROI-HIT reduces the over-exploration on the vast design space and advances the Pareto front more quickly. Consequently, it can obtain superior results within a constrained time budget. To further shorten the optimization time, ROI-HIT prunes unimportant variables via a sensitivity matrix and reduces the number of dimensions used for modeling and optimization. For time-consuming VLSI flow simulations, an asynchronous parallel strategy is employed.

## Usage
```
bash run.sh
```
Note: To run the code successfully, it is necessary to add the commands required for simulation in the `sim` function of `main.py`. For the experiments described in our paper, it is required to set up the [Chipyard](https://github.com/ucb-bar/chipyard) framework and utilize Cadence Genus tool.

## Requirement
```
numpy
torch
gpytorch
botorch
MiniSom
```

## Cite
```
@article{ROIHIT,
  title={ROI-HIT: Region of Interest-driven High-dimensional Microarchitecture Design Space Exploration},
  author={Zhao, Xuyang and Gao, Tiannig and Zhao, Aidong and Bi, Zhaori and Yan, Changhao and Yang, Fan and Wang, Sheng-Guo and Zhou, Dian and Zeng, Xuan},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2024},
  publisher={IEEE}
}
```
