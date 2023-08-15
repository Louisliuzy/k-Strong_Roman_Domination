
# k-Strong Roman Domination

This archive is distributed under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported in the paper 
[On the k-Strong Roman Domination Problem](https://www.sciencedirect.com/science/article/pii/S0166218X20302675) by Z. Liu et. al. The data generated for this study are included with the codes.

## Cite

To cite this repository, please cite the [paper](https://www.sciencedirect.com/science/article/pii/S0166218X20302675).

Below is the BibTex for citing the manuscript.

```
@article{Liu2020,
  title={On the k-Strong Roman Domination Problem},
  author={Liu, Zeyu and Li, Xueping and Khojandi, Anahita},
  journal={Discrete Applied Mathematics},
  volume={285},
  pages={227--241},
  year={2020},
  publisher={Elsevier}
}
```

## Description

The goal of this repository is to solve the k-strong Roman domination problem with L-shped method. Please refer to the manuscript for further details.

## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `gurobipy`;
3. `matplotlib`;
4. `networkx`.

## Usage

In `main`, there are three functions. `graph_generation()` generates testing instances. `ip_solver()` solves k-strong Roman domination with integer programming. `Lshaped()` solves k-strong Roman domination with Benders decomposition.


## Support

For support in using this software, submit an
[issue](https://github.com/Louisliuzy/k-Strong_Roman_Domination/issues).
