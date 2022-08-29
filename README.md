# SWAIN –––– a Semi-Distributed Rainfall-Runoff model

The following repo contains the code and tools to train several Rainfall-Runoff deep learning models based on graphs for the SWAIN project.

## Goal and data availability

The goal is to repesent the entire watershed of a given river (mainly the Danube river, focus of the LamaH-CE dataset) as a Graph representing gauge station as Nodes, the river geography as weighted Edges, and exploit exogenous information (such as weather data and/or catchment/basin information).

Gauge stations measure the flow rate (m^3/s) in various points of the river. Each one has a water basin associated, where weather measurements/forecasts have been collected. Moreover, there could be some basin-associated attributes, such as soil characteristics and environmental indicators.

The problem is thus centered on a one-step ahead non-linear regression task on historical time-series data, each timestamp being associated both to the measurements time-series (univariate, water flow) as well as the exougenous data time-series (multivariate).

Exogenous informations are heterogenous data; these influence the model prediction outcome, but viceversa do not get affected by the model itself. These are important to make sensible predictions w.r.t. real world conditions.

The following datasets have been implemented:

- Danube river, LamaH-CE dataset (basin delineation B)
- Ergene river

## Structure and Code base

This repo is based on the [tsl (Torch Spatiotemporal)](https://github.com/TorchSpatiotemporal/tsl) framework by [Andrea Cini](https://github.com/andreacini) and [Ivan Marisca](https://github.com/marshka).

The repository is structured as follows:

- **dataset**: dataset classes for different data corpora 
- **models**: PyTorch models
- **scripts**: running the model trainer, running the results plotter
- **utils**: model evaluator, results plotter, and loss metrics
- **tsl_config**: model configurations
- **default_config.yaml**: TSL setup (Neptune token and account included)

The following models have been implemented:

- DCRNN
- GAT
- GraphWaveNET
- GatedGN

## How to
In order to run a particular model, execute the following command:

`python -m scripts.run_gnn --model-name=gat --config=gat.yaml`

Each experiment is automatically logged, unless specified otherwise, on Neptune. It keeps track of run configurations and some metrics.

**The core output is saved on the machine that runs the experiments!**

This is crucial to run the interactive virtualization built upon Bokeh. In order to see results from a particular run, execute the following command:

`python -m scripts.run_plotter --dataset=lamah --exp-path=./logs/LMHRR-73`

## References

LamaH-CE dataset (Klingler, C. and Schulz, K. and Herrnegger, M., [2021](https://essd.copernicus.org/articles/13/4529/2021/))