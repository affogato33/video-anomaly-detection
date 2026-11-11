# video-anomaly-detection

This is a Video anomaly detection project using pretrained ResNet18 features and a ConvLSTM predictor. The model learns to predict future feature representations and uses prediction errors to identify any unusual events in video sequences.

## About

The main goal is to learn how normal video sequences behave and to use differences between predicted and actual features to detect any unusual activity.

This project is tested on two datasets:

- **UCSD Ped2**
- **CUHK Avenue**

ConvLSTM model is compared with PatchCore and other temporal baselines. I also worked with different sequence lengths, prediction horizons, camera motion and other disturbances to see how the model behaves under different conditions.

## Main results

| Dataset     | Model              | Frame AUROC |
| ----------- | ------------------ | ----------: |
| UCSD Ped2   | ConvLSTM           |      0.7693 |
| UCSD Ped2   | Spatial + Temporal |      0.7458 |
| CUHK Avenue | ConvLSTM           |      0.8072 |
| CUHK Avenue | Spatial + Temporal |      0.8877 |

Spatial and temporal combination works better on Avenue whereas the ConvLSTM alone performs better on Ped2.

For pixel level localisation, the spatial and temporal model achieves an AUROC of 0.9792 and an AUPRO of 0.9138 on Ped2.

## Experiments

This project contains:

- Comparison with patchcore and temporal baselines
- Experiments with different input sequence lengths
- Prediction horizon experiments
- Camera motion and other disturbance tests
- Pixel level anomaly localisation
- Five seed experiments and video level bootstrap analysis
- Streaming inference and ONNX Runtime benchmarking

## Inference

Streaming pipeline was benchmarked on NVIDIA Tesla T4.

- throughput: 64.31 FPS
- median latency: 15.41 ms
- peak gpu memory: 884.9 MB

ONNX model was also compared with pytorch to check its consistency and inference speed.

## Running the project

1. clone the repository.
2. install the python packages.
3. open the jupyter notebook and run the code.

The code automatically downloads UCSD Ped2 and CUHK Avenue if they are not there. PS: Internet access is needed for the downloads. Dataset paths can also be configured if you already have the datasets.


## Limitations

The model is sensitive to camera motion and changes in frame sampling. Longer input sequences also do not show a large improvement over shorter ones.

Performance also varies across datasets and adding spatial and temporal scores may not always improve the results.
