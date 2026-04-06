# TimeRFT: Stimulating Generalizable Time Series Forecasting for TSFMs via Reinforcement Finetuning

TimeRFT is one of the first poineering work that enhances the forecasting accuracy and generalization of TSFM finetuning by time series reinforcement learning. We acknowledge that our finetuning experiments are built upon the pretrained MOIRAI-MoE model from [uni2ts](https://github.com/SalesforceAIResearch/uni2ts).

## Quickstart
1. Data Preparation

You can download the real-world time series datasets from [fev-bench](https://huggingface.co/datasets/autogluon/fev_datasets). Then transform the obtained Huggingface dataset into the suitable train/val/test form by ```shell python data_converter.py```

2. Experiment Configuration

All RFT-based or SFT-based TSFM adaptation experiments across various forecasting tasks (i.e. univariate, multivariate and covariate-informed forecasting) and training data regimes (e.g. 20% few-shot and 100% full-shot) can be conducted by curating the .yaml file under the folder `configs/`.

3. Training and Evaluation

For univariate and multivariate forecasting tasks, you can implement TimeRFT by ```shell python finetune.py``` and ```shell python test.py```, with modifying "cfg_path".

For covariate-informed forecasting tasks, you can implement TimeRFT by ```shell python finetune_cov.py``` and ```shell python test_cov.py```, with modifying "cfg_path" and "target_dim_pred".
