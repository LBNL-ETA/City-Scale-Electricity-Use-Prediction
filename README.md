# City-Scale Electricity Use Prediction

This is the official repository that implements the following paper:

> *Zhe Wang, Han Li, Tianzhen Hong, Mary Ann Piette. 2021. Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models. Submitted to Advance in Applied Energy*

[[paper_submitted]](docs/paper_submitted.pdf)[[paper_online]](https://doi.org/10.1016/j.adapen.2021.100025)

# Overview
We developed data-driven models to predict city-scale electricity consumption.
- We developed and compared four models: (1) five parameter change-point model, (2) Heating/Cooling Degree Hour model, (3) time series decomposed model implemented by Facebook Prophet, (4) Gradient Boosting Machine implemented by Microsoft lightGBM, and (5) three widely-used machine learning models (Random Forest, Support Vector Machine, Neural Network).
- We applied our models to explore how extreme weather events (e.g., heat waves) and unexpected public health events (e.g. COVID-19 pandemic) influenced each city’s electricity demand

<img src="docs/overview.jpeg" width="1000" />


# Code Usage
### Clone repository
```
git clone https://github.com/LBNL-ETA/City-Scale-Electricity-Use-Prediction
cd City-Scale-Electricity-Use-Prediction
```

### Set up the environment 
Set up the virtual environment with your preferred environment/package manager.

The instruction here is based on **conda**. ([Install conda](https://docs.anaconda.com/anaconda/install/))
```
conda create --name cityEleEnv python=3.8 -c conda-forge -f requirements.txt
conda activate cityEleEnv
```

### Repository structure
``bin``: Runnable programs, including Python scripts and Jupyter Notebooks

``data``: Raw data, including city-level electricity consumption and weather data

``docs``: Manuscript submitted version

``results``: Cleaned-up data, generated figures and tables


### Running
You can replicate our experiments, generate figures and tables used in the manuscript using the Jupyter notebooks saved in ``bin``: `section3.1 EDA.ipynb`, `section3.2 linear model.ipynb`, `section3.3 time-series model.ipynb`, `section3.4 tabular data model.ipynb`, `section4.1 model comparison.ipynb`, `section4.2 heat wave.ipynb`, `section4.3 convid.ipynb`

*Notes.*
- Official Documentation of [Facebook Prophet](https://facebook.github.io/prophet/).
- Official Documentation of [Microsoft lightGBM](https://github.com/Microsoft/LightGBM). 

### Feedback

Feel free to send any questions/feedback to: [Zhe Wang](mailto:zwang5@lbl.gov ) or [Tianzhen Hong](mailto:thong@lbl.gov)

### Citation

If you use our code, please cite us as follows:

```
Wang, Z., Hong, T., Li, H. and Piette, M.A., 2021. Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models. Advances in Applied Energy, p.100025.

@article{wang2021predicting,
  title={Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models},
  author={Wang, Zhe and Hong, Tianzhen and Li, Han and Piette, Mary Ann},
  journal={Advances in Applied Energy},
  pages={100025},
  year={2021},
  publisher={Elsevier}
}
```