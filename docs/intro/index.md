# Introduction

## Concept

The Integrated FLIMAnalyzer: It is publicly available software that we developed in Python. The
code is maintained in a public GitHub repository, https://github.com/uvaKCCI/flimanalyzer . The
software provides tools to import from csv files raw data containing fitted FLIM data along with tools
to filter, sort, rearrange, and aggregate data in spreadsheet format. The imported data can further be
analyzed with well established statistical methods, including PCA and Kolmogorov–Smirnov (KS)
statistics provided through common scikit-learn Python packages. Data can be visualized in custom
plots based on third-party pandas, matplotlib, and seaborn packages. The autoencoders are
implemented using the PyTorch framework, and the software provides tools for hyperparameter
tuning of the developed data augmentation and feature extraction encoders. 

Additional details, key concepts and user guide are documented here, https://flimanalyzer.readthedocs.io/
