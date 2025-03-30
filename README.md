# Laminoboost 

Memoization-based Scalable Laminography Reconstruction.

## Convergence
![Figure 1: Convergence](./images/curve.png)

<p align="center"><b>Figure 1: Convergence rate</b> 

 The convergence was tested with the dataset (1k,1k,1k) and $\tau$ = 0.95. We have two observations: 1. No additional iteration for convergence is required. 2. The methods with memozation and without memoization have the same convergence rate over iterations.


## Bandwidth 
![Figure 2: Bandwidth](./images/bandwidth_vs_node_number.png)

<p align="center"><b>Figure 2: Scalability test with different number of nodes</b> 

The bandwidth was tested with the dataset (1k,1k,1k) reconstruction and 1-16 compute nodes connected by slingshot11(Max Bandwidth 200Gbps). Each compute node is equipped with a single AMD EPYC 7543P processor featuring 32 Zen3 cores (64 hardware threads) operating at 2.8 GHz. 


## Latency distribution
![Figure 3: Latency](./images/latency.png)

<p align="center"><b>Figure 3: Latency distribution</b> 

The remote memoization latency distribution was tested with the dataset (1k,1k,1k). We tested 1-16 compute nodes performing remote memoization query simultaneously on one memory node. The nodes are connected by slingshot11(Max Bandwidth 200Gbps). 
