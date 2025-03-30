# Laminoboost 

A computational laminography reconstruction framework with remote memoization.


## Convergence
![Convergence](./images/curve.png)

The convergence is tested with the (1k,1k,1k) dataset and tau = 0.95. No additional iteration for convergence is required.


## Bandwidth 
![Bandwidth](./images/bandwidth_vs_node_number.png).

The bandwidth is test with (1k,1k,1k) dataset reconstruction and 1-16 nodes connected by slingshot11. Each node is equipped with a single AMD EPYC 7543P processor featuring 32 Zen3 cores (64 hardware threads) operating at 2.8 GHz. 






