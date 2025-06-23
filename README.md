This repository contains the source code for the TANDEM-DIMPLE project. 

Github repository: https://github.com/locitran/tandem-dimple.git. 

TANDEM-DIMPLE is a DNN model designed to predict the pathogenicity of missense variants. The model is trained on R20000 set obtained from the [Rhapsody](https://academic.oup.com/bioinformatics/article/36/10/3084/5758260?login=true) study, using a wide range of features, including sequence&chemical, structural, and dynamics features.

![Summary of Results](docs/images/result_summary.png)
**Comparison of prediction accuracy across general and disease-specific models.**
(A) Schematic overview of training and evaluation pipelines. 20,361 SAVs from Rhapsody (Ponzoni et al., 2020), referred to as R20000, were split into R20000<sub>train</sub>, R20000<sub>val</sub>, and R20000<sub>test</sub> in a 72:18:10 ratio. DNN-based general disease models (left) were trained on R20000<sub>train</sub> using either the TANDEM or Rhapsody feature sets. Model evaluation was conducted on R20000<sub>test</sub>, along with two additional SAV sets related to specific diseases: GJB2<sub>knw</sub> and RYR1<sub>knw</sub>. Subsequently, 30% of the SAVs from each set (GJB2<sub>train</sub>/RYR1<sub>train</sub>) were applied to fine-tune TANDEM models via transfer learning, while 10% (GJB2<sub>test</sub>/RYR1<sub>test</sub>) served in evaluating the specific disease models. (B-C) Comparison of prediction accuracy on independent test sets using general disease models and specific disease models.

This repository contains:
1. The code to produce the features
2. [TANDEM-DIMPLE model](models/different_number_of_layers/20250423-1234-tandem/n_hidden-5)
3. Transfer-learned model for two specific diseases: [GJB2](models/transfer_learning_GJB2) and [RYR1](models/TransferLearning_RYR1).

To install the code, please follow this [instruction](docs/installation.md).

Input format and output format are described in the [input_output_format.md](docs/input_output_format.md) file.


```bibtex
@article{Loci2025,
  author  = {Loci Tran, Chen-Hua Lu, Pei-Lung Chen, Lee-Wei Yang},
  journal = {Bioarchiv},
  title   = {Predicting the pathogenicity of SAVs Transfer-leArNing-ready and Dynamics-Empowered Model for DIsease-specific Missense Pathogenicity Level Estimation},
  year    = {2025},
  volume  = {*.*},
  number  = {*.*},
  pages   = {*.*},
  doi     = {*.*}
}
```   