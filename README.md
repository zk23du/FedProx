# FedAvg
<h2> Image </h2>
<h3> Dataset = Cifar10 </h3>
<h3> Epochs = 300 </h3>
<h3> Clients = 5 </h3>
Optimizers: DFW, PerturbedGradientDescent (specifically for FedProx)

PGD performed better (FedProx), then DFW with FedProx and DFW FedAvg are almost same!!
<h4> ACCURACY</h4>

| Optimizer | 2 | 4 | 6 | 8 | 16 | --- |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 70.0573 | 70.7439 | 69.4774 | 69.5041 | 67.1977 | FedAvg |
| BCDFW | 71.8571 | 66.2378 | 64.6114 | 61.2518 | 54.4661 | FedAvg |
| DFW | 80.0069 | 78.6828 | 78.1630 | 77.9363 | 77.5963 |  FedAvg|
| DFW | 79.7160 | 78.6162 | 78.6495 | 78.6695 | 78.8961 | FedProx |
| PGD | 82.6956 | 82.1157 | 81.50 | 81 | 79.7560 | FedProx |

<h4> LOSS</h4>

| Optimizer | 2 | 4 | 6 | 8 | 16 | --- |
| --- | --- | --- | --- | --- | --- | --- |
| SGD | 0.4813 | 0.6934 | 0.9535 | 0.6434 | 0.8552 | FedAvg  |
| BCDFW | 0.9481 | 0.9484 | 0.9879 | 1.0980 | 1.2994 | FedAvg  |
| DFW | 0.0945 | 0.1713 | 0.1977 | 0.2131 | 0.2879 | FedAvg  |
| DFW | 0.1136 | 0.1769 | 0.1730 | 0.2230 | 0.2983 | FedProx |
| PGD | 0.0145 | 0.0134 | 0.0799 | 0.0761 | 0.3226 | FedProx |

<h2> TEXT </h2>
<h3> Dataset = Agnews </h3>
<h3> Epochs = 100 </h3>
<h3> Clients = 5 </h3>

<h4> ACCURACY</h4>

| Optimizer | 2 | 4 | 6 | 8 | 16 | Model | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SGD | 86.3457 | 83.3321 | 79.5954 | 75.4595| 58.5064 |TextCNN | FedAvg |
| BCDFW | 86.0398 | 81.5308 | 78.0943 | 74.5810 | 68.3397 | TextCNN | FedAvg |
| DFW | 87.7052 | 85.7992 | 84.8755 | 83.7103 | 80.0228 | TextCNN | FedAvg |
| DFW |  |  |  |  |  | TextCNN | FedProx |
| PG |  |  |  |  |  | TextCNN | FedProx |
| SGD | 55.5993 | 42.8730 | 38.9105 | 34.8520 | 31.9903 | fastText | FedAvg |
| BCDFW | 87.4970 | 82.2310 | 77.7310 | 73.41132 | 59.6535 | fastText | FedAvg |
| DFW | 94.5004 | 92.8077 | 92.0208 | 91.2967 | 89.2810 | fastText | FedAvg |
| DFW |  |  |  |  |  | fastText | FedProx |
| PG |  |  |  |  |  | TextCNN | FedProx |

<h4> LOSS</h4>

| Optimizer | 2 | 4 | 6 | 8 | 16 | Model |
| --- | --- | --- | --- | --- | --- | --- |
| SGD| 0.4458 | 0.5387 | 0.6486 | 0.7611 | 1.1765 | TextCNN |
| BCDFW | 0.4811 | 0.5967 | 0.6760 | 0.7697  | 0.9463 | TextCNN |
| DFW | 0.3911 | 0.4447 | 0.4776 | 0.5117 | 0.6162 | TextCNN |
| DFW |  |  |  |  |  | TextCNN | FedProx |
| PG |  |  |  |  |  | TextCNN | FedProx |
| SGD | 1.0565 | 1.2664 | 1.3303 | 1.3573 | 1.3817 | fastText |  
| BCDFW | 0.3782 | 0.4809 | 0.5856 | 0.6884 | 0.9822 | fastText |
| DFW | 0.2039 | 0.2458 | 0.2722 | 0.2795 | 0.3305 | fastText |
| DFW |  |  |  |  |  | fastText | FedProx |
| PG |  |  |  |  |  | TextCNN | FedProx |
